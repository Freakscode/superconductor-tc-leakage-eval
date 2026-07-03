"""
get_datasetA.py -- download the UCI Superconductivity Data Set (Hamidieh 2018, #464).
Writes train.csv (21,263 x 82) and unique_m.csv into ../data/.
The raw files are ~28 MB and are NOT redistributed in this repo; this script
fetches them from the canonical UCI archive so results are reproducible.
"""
import io, os, sys, zipfile, urllib.request

URL = "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")

def main():
    os.makedirs(OUT, exist_ok=True)
    print("downloading", URL)
    raw = urllib.request.urlopen(URL, timeout=120).read()
    print(f"  {len(raw)} bytes")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in ("train.csv", "unique_m.csv"):
            if name in z.namelist():
                z.extract(name, OUT)
                print("  extracted", os.path.join(OUT, name))
    print("done. Dataset A ready in", os.path.abspath(OUT))

if __name__ == "__main__":
    main()
