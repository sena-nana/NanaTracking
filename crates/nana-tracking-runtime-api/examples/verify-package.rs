use std::{env, path::Path};

use nana_tracking_runtime_api::verify_model_package;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .ok_or("usage: verify-package <model-package>")?;
    let package = verify_model_package(Path::new(&path))?;
    println!(
        "{} {} {}",
        package.metadata.model_family,
        package.metadata.model_version,
        package.metadata.guaranteed_profile
    );
    Ok(())
}
