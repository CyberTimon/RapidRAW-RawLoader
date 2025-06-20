use std::path::Path;
use image::{ImageBuffer, RgbImage};
use rayon::prelude::*;
use std::io::Cursor; // Needed for in-memory reading

// --- Algorithm Selection ---
// Choose which demosaicing algorithm to use.
enum DemosaicAlgorithm {
    Linear,
    Menon,
}

// --- Bayer Pattern Definition ---
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BayerPattern {
    RGGB, BGGR, GRBG, GBRG,
}

// --- Menon Algorithm Helpers ---
// Helper for 1D convolution axis
enum Axis {
    Horizontal,
    Vertical,
}
// Bilinear interpolation kernel for color channels
const K_B: [f32; 3] = [0.5, 0.0, 0.5];


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- Configuration ---
    let input_path = "sample.arw"; // Change to your RAW file (e.g., .arw, .dng, .cr2)
    let output_path = "output.png";
    let algorithm_to_use = DemosaicAlgorithm::Menon; // <-- CHANGE ALGORITHM HERE
    let use_menon_refining_step = true; // Only affects the Menon algorithm
    let exposure_compensation = 1.5;

    // --- 1. Load RAW File and Metadata ---
    println!("Reading file into memory: {}", input_path);
    // Read the whole file into a byte buffer first. This allows us to pass it to multiple parsers.
    let file_bytes = std::fs::read(Path::new(input_path))?;

    println!("Loading RAW data from buffer...");
    let raw_image = rawloader::decode(&mut Cursor::new(&file_bytes))?;
    let data = match raw_image.data {
        rawloader::RawImageData::Integer(data) => data,
        _ => panic!("This example only supports integer raw data."),
    };

    println!("--- Full EXIF Data (from kamadak-exif) ---");
    let exif_reader = exif::Reader::new();
    match exif_reader.read_from_container(&mut Cursor::new(&file_bytes)) {
        Ok(exif) => {
            if exif.fields().len() == 0 {
                println!("  No EXIF data found in file.");
            } else {
                for field in exif.fields() {
                    println!("  {:<25}: {}",
                        field.tag,
                        field.display_value().with_unit(&exif)
                    );
                }
            }
        }
        Err(e) => {
            println!("  Could not read EXIF data: {}", e);
        }
    }
    println!("------------------------------------------\n");


    let width = raw_image.width as u32;
    let height = raw_image.height as u32;
    let bayer_pattern = match raw_image.cfa.to_string().as_str() {
        "RGGB" => BayerPattern::RGGB,
        "BGGR" => BayerPattern::BGGR,
        "GRBG" => BayerPattern::GRBG,
        "GBRG" => BayerPattern::GBRG,
        p => {
            println!("Unknown CFA pattern '{}', defaulting to RGGB", p);
            BayerPattern::RGGB
        }
    };

    // --- 2. Prepare Post-Processing Parameters ---
    let wb_coeffs = raw_image.wb_coeffs;
    let final_wb_coeffs = if wb_coeffs[1].abs() > 0.0001 {
        [wb_coeffs[0] / wb_coeffs[1], 1.0, wb_coeffs[2] / wb_coeffs[1]]
    } else {
        [1.0, 1.0, 1.0]
    };

    let max_value = raw_image.whitelevels[0] as f32;
    let black_level = raw_image.blacklevels[0] as f32;
    let dynamic_range = (max_value - black_level).max(1.0);

    // --- 3. Demosaic and Process Image ---
    let mut img_buffer: RgbImage = ImageBuffer::new(width, height);

    match algorithm_to_use {
        DemosaicAlgorithm::Linear => {
            println!("Processing {}x{} image with Linear Demosaicing...", width, height);
            let buffer = img_buffer.as_mut();

            // Process rows in parallel. Demosaicing and post-processing are done per-pixel.
            buffer.par_chunks_mut((width * 3) as usize)
                .enumerate()
                .for_each(|(y, row)| {
                    for x in 0..width {
                        // Step 1: Demosaic a single pixel
                        let (r_raw, g_raw, b_raw) = demosaic_pixel_optimized_linear(
                            &data, x, y as u32, width, height, bayer_pattern
                        );

                        // Step 2: Post-process the pixel
                        let r_bl = (r_raw - black_level).max(0.0);
                        let g_bl = (g_raw - black_level).max(0.0);
                        let b_bl = (b_raw - black_level).max(0.0);

                        let r_wb = r_bl * final_wb_coeffs[0];
                        let g_wb = g_bl * final_wb_coeffs[1];
                        let b_wb = b_bl * final_wb_coeffs[2];

                        let r_norm = ((r_wb / dynamic_range) * exposure_compensation).clamp(0.0, 1.0);
                        let g_norm = ((g_wb / dynamic_range) * exposure_compensation).clamp(0.0, 1.0);
                        let b_norm = ((b_wb / dynamic_range) * exposure_compensation).clamp(0.0, 1.0);

                        let r_gamma = r_norm.powf(1.0 / 2.2);
                        let g_gamma = g_norm.powf(1.0 / 2.2);
                        let b_gamma = b_norm.powf(1.0 / 2.2);

                        // Step 3: Write to output buffer
                        let base = (x * 3) as usize;
                        row[base + 0] = (r_gamma * 255.0) as u8;
                        row[base + 1] = (g_gamma * 255.0) as u8;
                        row[base + 2] = (b_gamma * 255.0) as u8;
                    }
                });
        }
        DemosaicAlgorithm::Menon => {
            println!("Processing {}x{} image with Menon (2007) Demosaicing...", width, height);
            
            // Step 1: Demosaic the entire image first. This is a whole-image operation.
            let rgb_f32_data = demosaic_menon2007(&data, width, height, bayer_pattern, use_menon_refining_step);

            println!("Applying post-processing (White Balance, Gamma)...");
            let buffer = img_buffer.as_mut();

            // Step 2: Apply post-processing in parallel per row.
            buffer.par_chunks_mut((width * 3) as usize)
                .enumerate()
                .for_each(|(y, row_out)| {
                    for x in 0..width {
                        let idx = (y * width as usize + x as usize) * 3;
                        let r_raw = rgb_f32_data[idx];
                        let g_raw = rgb_f32_data[idx + 1];
                        let b_raw = rgb_f32_data[idx + 2];

                        let r_bl = (r_raw - black_level).max(0.0);
                        let g_bl = (g_raw - black_level).max(0.0);
                        let b_bl = (b_raw - black_level).max(0.0);

                        let r_wb = r_bl * final_wb_coeffs[0];
                        let g_wb = g_bl * final_wb_coeffs[1];
                        let b_wb = b_bl * final_wb_coeffs[2];

                        let r_norm = ((r_wb / dynamic_range) * exposure_compensation).clamp(0.0, 1.0);
                        let g_norm = ((g_wb / dynamic_range) * exposure_compensation).clamp(0.0, 1.0);
                        let b_norm = ((b_wb / dynamic_range) * exposure_compensation).clamp(0.0, 1.0);

                        let r_gamma = r_norm.powf(1.0 / 2.2);
                        let g_gamma = g_norm.powf(1.0 / 2.2);
                        let b_gamma = b_norm.powf(1.0 / 2.2);

                        let base = (x * 3) as usize;
                        row_out[base + 0] = (r_gamma * 255.0) as u8;
                        row_out[base + 1] = (g_gamma * 255.0) as u8;
                        row_out[base + 2] = (b_gamma * 255.0) as u8;
                    }
                });
        }
    }

    // --- 4. Save Final Image ---
    println!("Saving final image to: {}", output_path);
    img_buffer.save(output_path)?;
    println!("Conversion completed successfully!");

    Ok(())
}


// #############################################################################
// ALGORITHM 1: LINEAR INTERPOLATION
// #############################################################################

fn demosaic_pixel_optimized_linear(
    raw_data: &[u16], x: u32, y: u32, width: u32, height: u32,
    pattern: BayerPattern
) -> (f32, f32, f32) {
    let x_i = x as i32;
    let y_i = y as i32;

    let get_raw = |px: i32, py: i32| -> f32 {
        let clamped_x = px.max(0).min(width as i32 - 1) as u32;
        let clamped_y = py.max(0).min(height as i32 - 1) as u32;
        raw_data[(clamped_y * width + clamped_x) as usize] as f32
    };

    let get_color_at = |px: i32, py: i32| -> char {
        let (is_red_row, is_red_col) = match pattern {
            BayerPattern::RGGB => (py % 2 == 0, px % 2 == 0),
            BayerPattern::BGGR => (py % 2 == 1, px % 2 == 1),
            BayerPattern::GRBG => (py % 2 == 0, px % 2 == 1),
            BayerPattern::GBRG => (py % 2 == 1, px % 2 == 0),
        };
        if is_red_row == is_red_col {
            if is_red_row { 'R' } else { 'B' }
        } else {
            'G'
        }
    };

    match get_color_at(x_i, y_i) {
        'R' => {
            let r = get_raw(x_i, y_i);
            let b = (get_raw(x_i - 1, y_i - 1) + get_raw(x_i + 1, y_i - 1) +
                     get_raw(x_i - 1, y_i + 1) + get_raw(x_i + 1, y_i + 1)) / 4.0;
            let g_n = get_raw(x_i, y_i - 1);
            let g_s = get_raw(x_i, y_i + 1);
            let g_w = get_raw(x_i - 1, y_i);
            let g_e = get_raw(x_i + 1, y_i);
            let grad_v = (g_n - g_s).abs();
            let grad_h = (g_w - g_e).abs();
            let g = if grad_v < grad_h {
                (g_n + g_s) / 2.0
            } else if grad_h < grad_v {
                (g_w + g_e) / 2.0
            } else {
                (g_n + g_s + g_w + g_e) / 4.0
            };
            (r, g, b)
        }
        'B' => {
            let b = get_raw(x_i, y_i);
            let r = (get_raw(x_i - 1, y_i - 1) + get_raw(x_i + 1, y_i - 1) +
                     get_raw(x_i - 1, y_i + 1) + get_raw(x_i + 1, y_i + 1)) / 4.0;
            let g_n = get_raw(x_i, y_i - 1);
            let g_s = get_raw(x_i, y_i + 1);
            let g_w = get_raw(x_i - 1, y_i);
            let g_e = get_raw(x_i + 1, y_i);
            let grad_v = (g_n - g_s).abs();
            let grad_h = (g_w - g_e).abs();
            let g = if grad_v < grad_h {
                (g_n + g_s) / 2.0
            } else if grad_h < grad_v {
                (g_w + g_e) / 2.0
            } else {
                (g_n + g_s + g_w + g_e) / 4.0
            };
            (r, g, b)
        }
        'G' => {
            let g = get_raw(x_i, y_i);
            let (r, b) = if get_color_at(x_i + 1, y_i) == 'R' {
                let r_val = (get_raw(x_i - 1, y_i) + get_raw(x_i + 1, y_i)) / 2.0;
                let b_val = (get_raw(x_i, y_i - 1) + get_raw(x_i, y_i + 1)) / 2.0;
                (r_val, b_val)
            } else {
                let b_val = (get_raw(x_i - 1, y_i) + get_raw(x_i + 1, y_i)) / 2.0;
                let r_val = (get_raw(x_i, y_i - 1) + get_raw(x_i, y_i + 1)) / 2.0;
                (r_val, b_val)
            };
            (r, g, b)
        }
        _ => (0.0, 0.0, 0.0),
    }
}


// #############################################################################
// ALGORITHM 2: MENON (2007) DDFAPD
// #############################################################################

/// Demosaics a Bayer CFA image using the Menon (2007) DDFAPD algorithm.
#[allow(non_snake_case)]
fn demosaic_menon2007(
    cfa_data: &[u16],
    width: u32,
    height: u32,
    pattern: BayerPattern,
    use_refining_step: bool,
) -> Vec<f32> {
    let size = (width * height) as usize;
    let cfa: Vec<f32> = cfa_data.iter().map(|&p| p as f32).collect();

    // Create R, G, B masks
    let (R_m, G_m, B_m) = get_bayer_masks(width, height, pattern);

    // Separate CFA into R, G, B channels (sparse arrays)
    let mut R: Vec<f32> = vec![0.0; size];
    let mut G: Vec<f32> = vec![0.0; size];
    let mut B: Vec<f32> = vec![0.0; size];
    for i in 0..size {
        if R_m[i] { R[i] = cfa[i]; }
        if G_m[i] { G[i] = cfa[i]; }
        if B_m[i] { B[i] = cfa[i]; }
    }

    // --- Section II-A: Directional Green Interpolation ---
    let h_0 = [0.0, 0.5, 0.0, 0.5, 0.0];
    let h_1 = [-0.25, 0.0, 0.5, 0.0, -0.25];

    let G_H_conv: Vec<f32> = convolve_1d(&cfa, width, height, &h_0, Axis::Horizontal)
        .iter().zip(convolve_1d(&cfa, width, height, &h_1, Axis::Horizontal).iter())
        .map(|(a, b)| a + b).collect();
    let G_V_conv: Vec<f32> = convolve_1d(&cfa, width, height, &h_0, Axis::Vertical)
        .iter().zip(convolve_1d(&cfa, width, height, &h_1, Axis::Vertical).iter())
        .map(|(a, b)| a + b).collect();

    let G_H: Vec<f32> = G.iter().zip(G_m.iter()).zip(G_H_conv.iter())
        .map(|((&g, &mask), &conv)| if mask { g } else { conv }).collect();
    let G_V: Vec<f32> = G.iter().zip(G_m.iter()).zip(G_V_conv.iter())
        .map(|((&g, &mask), &conv)| if mask { g } else { conv }).collect();

    // --- Section II-B: Decision ---
    let mut C_H = vec![0.0; size];
    let mut C_V = vec![0.0; size];
    for i in 0..size {
        if R_m[i] {
            C_H[i] = R[i] - G_H[i];
            C_V[i] = R[i] - G_V[i];
        } else if B_m[i] {
            C_H[i] = B[i] - G_H[i];
            C_V[i] = B[i] - G_V[i];
        }
    }

    let D_H: Vec<f32> = (0..size).map(|i| {
        let x = i % width as usize;
        let prev_idx = i.saturating_sub(if x >= 2 { 2 } else { x });
        (C_H[i] - C_H[prev_idx]).abs()
    }).collect();

    let D_V: Vec<f32> = (0..size).map(|i| {
        let y = i / width as usize;
        let prev_idx = i.saturating_sub(if y >= 2 { 2 * width as usize } else { y * width as usize });
        (C_V[i] - C_V[prev_idx]).abs()
    }).collect();

    let k_box_5x5: [f32; 25] = [
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let d_H = convolve_2d(&D_H, width, height, &k_box_5x5);
    let d_V = convolve_2d(&D_V, width, height, &k_box_5x5);

    let M: Vec<bool> = d_H.iter().zip(d_V.iter()).map(|(h, v)| h <= v).collect(); // true for Horizontal
    let mut G_final: Vec<f32> = G_H.iter().zip(G_V.iter()).zip(M.iter())
        .map(|((&gh, &gv), &m)| if m { gh } else { gv }).collect();
    for i in 0..size { if G_m[i] { G_final[i] = G[i]; } }

    // --- Section II-C: Red and Blue Interpolation ---
    let (R_r, _) = get_row_masks(width, height, &R_m, &B_m);
    
    let mut R_final = R.clone();
    let mut B_final = B.clone();

    // Interpolate R/B at G locations first. This is a bilinear interpolation of the color differences.
    let R_conv_at_G_h = convolve_1d(&R, width, height, &K_B, Axis::Horizontal);
    let G_conv_at_G_h = convolve_1d(&G_final, width, height, &K_B, Axis::Horizontal);
    let B_conv_at_G_h = convolve_1d(&B, width, height, &K_B, Axis::Horizontal);
    let R_conv_at_G_v = convolve_1d(&R, width, height, &K_B, Axis::Vertical);
    let G_conv_at_G_v = convolve_1d(&G_final, width, height, &K_B, Axis::Vertical);
    let B_conv_at_G_v = convolve_1d(&B, width, height, &K_B, Axis::Vertical);

    for i in 0..size {
        if G_m[i] {
            if R_r[i] { // G on a red row (e.g., R-G-R-G)
                R_final[i] = G_final[i] + R_conv_at_G_h[i] - G_conv_at_G_h[i];
                B_final[i] = G_final[i] + B_conv_at_G_v[i] - G_conv_at_G_v[i];
            } else { // G on a blue row (e.g., B-G-B-G)
                R_final[i] = G_final[i] + R_conv_at_G_v[i] - G_conv_at_G_v[i];
                B_final[i] = G_final[i] + B_conv_at_G_h[i] - G_conv_at_G_h[i];
            }
        }
    }

    // Perform new convolutions on the now-denser R_final and B_final channels.
    // The previous convolutions were on sparse data. Now that G locations are filled,
    // we can correctly interpolate R at B locations and B at R locations.
    let R_final_conv_h = convolve_1d(&R_final, width, height, &K_B, Axis::Horizontal);
    let B_final_conv_h = convolve_1d(&B_final, width, height, &K_B, Axis::Horizontal);
    let R_final_conv_v = convolve_1d(&R_final, width, height, &K_B, Axis::Vertical);
    let B_final_conv_v = convolve_1d(&B_final, width, height, &K_B, Axis::Vertical);

    // Interpolate R at B locations, and B at R locations using the newly convolved data.
    for i in 0..size {
        if B_m[i] { // Interpolate R at a B location
            let rb_diff = if M[i] { R_final_conv_h[i] - B_final_conv_h[i] } else { R_final_conv_v[i] - B_final_conv_v[i] };
            R_final[i] = B_final[i] + rb_diff;
        } else if R_m[i] { // Interpolate B at an R location
            let br_diff = if M[i] { B_final_conv_h[i] - R_final_conv_h[i] } else { B_final_conv_v[i] - R_final_conv_v[i] };
            B_final[i] = R_final[i] + br_diff;
        }
    }

    let (mut R_out, mut G_out, mut B_out) = (R_final, G_final, B_final);

    if use_refining_step {
        (R_out, G_out, B_out) = refining_step_menon2007(
            &R_out, &G_out, &B_out, &R_m, &G_m, &B_m, &M, width, height
        );
    }

    // Interleave R, G, B channels into a single buffer
    let mut rgb_f32_data = vec![0.0; size * 3];
    for i in 0..size {
        rgb_f32_data[i * 3] = R_out[i];
        rgb_f32_data[i * 3 + 1] = G_out[i];
        rgb_f32_data[i * 3 + 2] = B_out[i];
    }
    rgb_f32_data
}

/// Implements the refining step from Section III of the Menon (2007) paper.
#[allow(non_snake_case)]
fn refining_step_menon2007(
    R_in: &[f32], G_in: &[f32], B_in: &[f32],
    R_m: &[bool], G_m: &[bool], B_m: &[bool], M: &[bool],
    width: u32, height: u32
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let size = (width * height) as usize;
    let mut R = R_in.to_vec();
    let mut G = G_in.to_vec();
    let mut B = B_in.to_vec();
    let (R_r, _) = get_row_masks(width, height, R_m, B_m);
    let fir = [1.0/3.0, 1.0/3.0, 1.0/3.0];

    // 1) Update Green component
    let R_G: Vec<f32> = R.iter().zip(G.iter()).map(|(r, g)| r - g).collect();
    let B_G: Vec<f32> = B.iter().zip(G.iter()).map(|(b, g)| b - g).collect();
    let R_G_h = convolve_1d(&R_G, width, height, &fir, Axis::Horizontal);
    let R_G_v = convolve_1d(&R_G, width, height, &fir, Axis::Vertical);
    let B_G_h = convolve_1d(&B_G, width, height, &fir, Axis::Horizontal);
    let B_G_v = convolve_1d(&B_G, width, height, &fir, Axis::Vertical);

    for i in 0..size {
        if R_m[i] {
            G[i] = R[i] - if M[i] { R_G_h[i] } else { R_G_v[i] };
        } else if B_m[i] {
            G[i] = B[i] - if M[i] { B_G_h[i] } else { B_G_v[i] };
        }
    }

    // 2) Update R and B at Green locations
    let R_G: Vec<f32> = R.iter().zip(G.iter()).map(|(r, g)| r - g).collect();
    let B_G: Vec<f32> = B.iter().zip(G.iter()).map(|(b, g)| b - g).collect();
    let R_G_h = convolve_1d(&R_G, width, height, &K_B, Axis::Horizontal);
    let R_G_v = convolve_1d(&R_G, width, height, &K_B, Axis::Vertical);
    let B_G_h = convolve_1d(&B_G, width, height, &K_B, Axis::Horizontal);
    let B_G_v = convolve_1d(&B_G, width, height, &K_B, Axis::Vertical);

    for i in 0..size {
        if G_m[i] {
            if R_r[i] { // G on a red row
                R[i] = G[i] + R_G_h[i];
                B[i] = G[i] + B_G_v[i];
            } else { // G on a blue row
                R[i] = G[i] + R_G_v[i];
                B[i] = G[i] + B_G_h[i];
            }
        }
    }

    // 3) Update R at B locations and B at R locations
    let R_B: Vec<f32> = R.iter().zip(B.iter()).map(|(r, b)| r - b).collect();
    let R_B_h = convolve_1d(&R_B, width, height, &fir, Axis::Horizontal);
    let R_B_v = convolve_1d(&R_B, width, height, &fir, Axis::Vertical);

    for i in 0..size {
        if B_m[i] {
            R[i] = B[i] + if M[i] { R_B_h[i] } else { R_B_v[i] };
        } else if R_m[i] {
            B[i] = R[i] - if M[i] { R_B_h[i] } else { R_B_v[i] };
        }
    }

    (R, G, B)
}

// --- Menon Helper Functions ---

fn get_bayer_masks(width: u32, height: u32, pattern: BayerPattern) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
    let mut r_mask = vec![false; (width * height) as usize];
    let mut g_mask = vec![false; (width * height) as usize];
    let mut b_mask = vec![false; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let (row_even, col_even) = (y % 2 == 0, x % 2 == 0);
            let c = match pattern {
                BayerPattern::RGGB => if row_even { if col_even {'R'} else {'G'} } else { if col_even {'G'} else {'B'} },
                BayerPattern::BGGR => if row_even { if col_even {'B'} else {'G'} } else { if col_even {'G'} else {'R'} },
                BayerPattern::GRBG => if row_even { if col_even {'G'} else {'R'} } else { if col_even {'B'} else {'G'} },
                BayerPattern::GBRG => if row_even { if col_even {'G'} else {'B'} } else { if col_even {'R'} else {'G'} },
            };
            match c { 'R' => r_mask[idx] = true, 'G' => g_mask[idx] = true, 'B' => b_mask[idx] = true, _ => {} }
        }
    }
    (r_mask, g_mask, b_mask)
}

fn get_row_masks(width: u32, height: u32, r_m: &[bool], b_m: &[bool]) -> (Vec<bool>, Vec<bool>) {
    let mut r_rows = vec![false; height as usize];
    let mut b_rows = vec![false; height as usize];
    for y in 0..height {
        if r_m[(y * width) as usize] || r_m[(y * width + 1).min(width*height-1) as usize] { r_rows[y as usize] = true; }
        if b_m[(y * width) as usize] || b_m[(y * width + 1).min(width*height-1) as usize] { b_rows[y as usize] = true; }
    }
    let mut r_r = vec![false; (width * height) as usize];
    let mut b_r = vec![false; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if r_rows[y as usize] { r_r[idx] = true; }
            if b_rows[y as usize] { b_r[idx] = true; }
        }
    }
    (r_r, b_r)
}

fn convolve_1d(data: &[f32], width: u32, height: u32, kernel: &[f32], axis: Axis) -> Vec<f32> {
    let mut output = vec![0.0; data.len()];
    let k_len = kernel.len();
    let k_center = k_len / 2;
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for k in 0..k_len {
                let offset = k as i32 - k_center as i32;
                let (px, py) = match axis {
                    Axis::Horizontal => (x as i32 + offset, y as i32),
                    Axis::Vertical => (x as i32, y as i32 + offset),
                };
                let mirror_x = if px < 0 { -px } else if px >= width as i32 { 2 * (width as i32 - 1) - px } else { px } as u32;
                let mirror_y = if py < 0 { -py } else if py >= height as i32 { 2 * (height as i32 - 1) - py } else { py } as u32;
                let idx = (mirror_y * width + mirror_x) as usize;
                sum += data[idx] * kernel[k];
            }
            output[(y * width + x) as usize] = sum;
        }
    }
    output
}

fn convolve_2d(data: &[f32], width: u32, height: u32, kernel: &[f32]) -> Vec<f32> {
    let k_side = (kernel.len() as f32).sqrt() as usize;
    if k_side * k_side != kernel.len() { panic!("2D kernel must be square."); }
    let k_center = k_side / 2;
    let mut output = vec![0.0; data.len()];
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for ky in 0..k_side {
                for kx in 0..k_side {
                    let offset_x = kx as i32 - k_center as i32;
                    let offset_y = ky as i32 - k_center as i32;
                    let px = x as i32 + offset_x;
                    let py = y as i32 + offset_y;
                    let mirror_x = if px < 0 { -px } else if px >= width as i32 { 2 * (width as i32 - 1) - px } else { px } as u32;
                    let mirror_y = if py < 0 { -py } else if py >= height as i32 { 2 * (height as i32 - 1) - py } else { py } as u32;
                    let data_idx = (mirror_y * width + mirror_x) as usize;
                    let kernel_idx = ky * k_side + kx;
                    sum += data[data_idx] * kernel[kernel_idx];
                }
            }
            output[(y * width + x) as usize] = sum;
        }
    }
    output
}