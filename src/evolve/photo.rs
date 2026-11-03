use super::*;

const PREVIEW_W: u32 = 288;

pub struct Photo {
    pub name: String,
    pub image: image::RgbaImage,
    pub texture: Option<egui::TextureHandle>,
    pub(super) shown: Option<Prep>,
}

pub(super) fn cover_crop(img: &image::RgbaImage, aspect: f32) -> image::RgbaImage {
    let (w, h) = img.dimensions();
    let (cw, ch) = if w as f32 / h as f32 > aspect { (((h as f32) * aspect).round() as u32, h) } else { (w, ((w as f32) / aspect).round() as u32) };
    image::imageops::crop_imm(img, (w - cw.min(w)) / 2, (h - ch.min(h)) / 2, cw.min(w).max(1), ch.min(h).max(1)).to_image()
}

pub fn load_photo(path: &Path, aspect: f32) -> Result<Photo, String> {
    let reader = image::ImageReader::open(path).map_err(|e| e.to_string())?.with_guessed_format().map_err(|e| e.to_string())?;
    let mut dec = reader.into_decoder().map_err(|e| e.to_string())?;
    let orient = dec.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(dec).map_err(|e| e.to_string())?;
    img.apply_orientation(orient);
    let crop = cover_crop(&img.to_rgba8(), aspect);
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    Ok(Photo { name, image: crop, texture: None, shown: None })
}

pub fn preview(photo: &mut Photo, ctx: &egui::Context, prep: Prep, canvas_mm: f32) -> egui::TextureHandle {
    if let Some(t) = photo.texture.as_ref().filter(|_| photo.shown == Some(prep)) {
        return t.clone();
    }
    let (w, h) = photo.image.dimensions();
    let ph = ((PREVIEW_W as f32) * h as f32 / w as f32).round().max(1.0) as u32;
    let small = prepare(&image::imageops::resize(&photo.image, PREVIEW_W, ph, image::imageops::FilterType::Triangle), prep, canvas_mm / PREVIEW_W as f32);
    photo.shown = Some(prep);
    let ci = egui::ColorImage::from_rgba_unmultiplied([PREVIEW_W as usize, ph as usize], small.as_raw());
    let t = ctx.load_texture("evolve_photo", ci, egui::TextureOptions::LINEAR);
    photo.texture = Some(t.clone());
    t
}
