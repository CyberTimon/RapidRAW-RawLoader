use std::path::Path;
use image::{ImageBuffer, RgbImage};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BayerPattern {
    RGGB, BGGR, GRBG, GBRG,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input_path = "sample.dng";
    let output_path = "output.png";

    println!("Loading RAW file: {}", input_path);
    let raw_image = rawloader::decode_file(Path::new(input_path))?;
    let data = match raw_image.data {
        rawloader::RawImageData::Integer(data) => data,
        _ => panic!("This example only supports integer raw data."),
    };

    let width = raw_image.width as u32;
    let height = raw_image.height as u32;
    let bayer_pattern = match raw_image.cfa.to_string().as_str() {
        "RGGB" => BayerPattern::RGGB,
        "BGGR" => BayerPattern::BGGR,
        "GRBG" => BayerPattern::GRBG,
        "GBRG" => BayerPattern::GBRG,
        _ => {
            println!("Unknown CFA pattern, defaulting to RGGB");
            BayerPattern::RGGB
        }
    };

    let wb_coeffs = raw_image.wb_coeffs;
    let final_wb_coeffs = if wb_coeffs[1] > 0.0001 {
        [wb_coeffs[0] / wb_coeffs[1], 1.0, wb_coeffs[2] / wb_coeffs[1]]
    } else {
        [1.0, 1.0, 1.0]
    };

    let max_value = raw_image.whitelevels[0] as f32;
    let black_level = raw_image.blacklevels[0] as f32;
    let dynamic_range = (max_value - black_level).max(1.0);
    let exposure_compensation = 1.5;

    println!("Processing {}x{} image...", width, height);

    let mut img_buffer: RgbImage = ImageBuffer::new(width, height);
    let buffer = img_buffer.as_mut();

    // Process rows in parallel
    buffer.par_chunks_mut((width * 3) as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..width {
                let (r_raw, g_raw, b_raw) = demosaic_pixel_optimized_linear(
                    &data, x, y as u32, width, height, bayer_pattern
                );

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
                row[base + 0] = (r_gamma * 255.0) as u8;
                row[base + 1] = (g_gamma * 255.0) as u8;
                row[base + 2] = (b_gamma * 255.0) as u8;
            }
        });

    println!("Saving final image to: {}", output_path);
    img_buffer.save(output_path)?;
    println!("Conversion completed successfully!");

    Ok(())
}

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