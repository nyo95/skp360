# RAD AI360 Visualizer

This is a separate SketchUp extension. It does not modify or depend on `RAD AI360 Exporter`.

## One-time setup

1. Open a terminal in this folder and run `python -m pip install -r requirements.txt`.
2. For AI rendering, copy `config.example.json` to `config.json`, then fill in a Cloudflare API token with **Workers AI Write** and your Account ID. The local `config.json` is never included in the RBZ package. Environment variables `CF_ACCOUNT_ID` and `CF_API_TOKEN` remain available as an override.
3. Create a SketchUp Scene for the desired viewpoint. Names starting with `AI360_` are listed first.

If `python` is not the command for your Python installation, set `RAD_AI360_PYTHON` to the complete Python executable path and restart SketchUp.

## Use

Open **Extensions > RAD AI360 Visualizer > Create 360 Panorama Job**. First use **Stitch only**, then inspect `equirectangular_2to1.png` in a 360 viewer. Once face orientation is correct, select **Generate with Cloudflare FLUX.2 Klein 4B**.

Cloudflare limits FLUX.2 Klein 4B reference images to under 512px. The worker scales only the temporary AI reference down to 511px and asks for output at the cubemap face resolution (up to 1920px).

Each job writes its source faces, job settings, worker log, final panorama and report to the folder you choose. Generated imagery is for visualisation, not a guarantee of architectural accuracy.
