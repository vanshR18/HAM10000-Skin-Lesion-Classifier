

import sys
import json
import time
import argparse
import requests
from pathlib import Path

# ── Config 
DEFAULT_URL   = "http://localhost:8000"
PASS  = " PASS"
FAIL  = " FAIL"
SKIP  = " SKIP"
results = []

def log(status, name, detail=""):
    icon = {"PASS": "Pass", "FAIL": "Fail", "SKIP": "skipped "}.get(status, "•")
    msg  = f"  {icon}  {name}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    results.append((status, name))


def separator(title=""):
    line = "─" * 55
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


# ── Tests 
def test_root(base_url):
    separator("GET /")
    try:
        r = requests.get(f"{base_url}/", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert "message" in data
        log("PASS", "Root returns 200", data["message"])
        assert "endpoints" in data
        log("PASS", "Endpoints listed in response")
    except Exception as e:
        log("FAIL", "GET /", str(e))


def test_health(base_url):
    separator("GET /health")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        assert r.status_code == 200
        data = r.json()
        log("PASS", "Health returns 200")
        log("PASS" if data["status"] == "ok" else "FAIL",
            f"Status = '{data['status']}'")
        log("PASS" if data["model_loaded"] else "FAIL",
            f"Model loaded = {data['model_loaded']}")
        log("PASS", f"Device = {data['device']}")
        log("PASS", f"Classes = {data['num_classes']}")
        log("PASS", f"Uptime = {data['uptime_sec']:.1f}s")
    except Exception as e:
        log("FAIL", "GET /health", str(e))


def test_classes(base_url):
    separator("GET /classes")
    try:
        r = requests.get(f"{base_url}/classes", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data["num_classes"] == 7
        log("PASS", f"Returns 7 classes")
        for cls in data["classes"]:
            log("PASS", f"Class: {cls['key']:<8} | {cls['full_name'][:35]}")
    except Exception as e:
        log("FAIL", "GET /classes", str(e))


def test_predict_no_file(base_url):
    separator("POST /predict — Validation Tests")
    # No file
    try:
        r = requests.post(f"{base_url}/predict", timeout=10)
        assert r.status_code == 422
        log("PASS", "No file → 422 Unprocessable Entity")
    except Exception as e:
        log("FAIL", "No file validation", str(e))

    # Wrong file type
    try:
        r = requests.post(
            f"{base_url}/predict",
            files={"file": ("test.txt", b"hello world", "text/plain")}
        )
        assert r.status_code == 400
        log("PASS", "Wrong file type → 400 Bad Request")
    except Exception as e:
        log("FAIL", "Wrong file type validation", str(e))


def test_predict_with_image(base_url, image_path: str):
    separator("POST /predict — Single Image")

    if not image_path or not Path(image_path).exists():
        # Create a synthetic test image if no path given
        try:
            from PIL import Image as PILImage
            import io
            img = PILImage.new("RGB", (224, 224), color=(180, 120, 100))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()
            filename  = "synthetic_test.jpg"
            log("PASS", "No image path provided — using synthetic 224×224 test image")
        except ImportError:
            log("SKIP", "Pillow not installed — skipping image prediction test")
            return
    else:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        filename = Path(image_path).name
        log("PASS", f"Using image: {filename}")

    try:
        start = time.time()
        r = requests.post(
            f"{base_url}/predict",
            files={"file": (filename, img_bytes, "image/jpeg")},
            timeout=30
        )
        elapsed = (time.time() - start) * 1000

        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        data = r.json()

        log("PASS", f"Predict returns 200")
        log("PASS", f"Predicted class    : {data['predicted_class']} ({data['full_name']})")
        log("PASS", f"Confidence         : {data['confidence_pct']}")
        log("PASS", f"Is malignant       : {data['is_malignant']}")
        log("PASS", f"Inference time     : {data['inference_time_ms']} ms")
        log("PASS", f"Total request time : {elapsed:.0f} ms")
        log("PASS" if len(data["all_probabilities"]) == 7 else "FAIL",
            f"All 7 class probs returned ({len(data['all_probabilities'])} found)")

        print("\n   All class probabilities:")
        for cls in data["all_probabilities"]:
            bar = "█" * int(float(cls["probability"]) * 40)
            print(f"    {cls['class_key']:<8} {cls['percentage']:>7}  {bar}")

    except Exception as e:
        log("FAIL", "POST /predict", str(e))


def test_batch_predict(base_url, image_path: str):
    separator("POST /predict/batch — Batch Test")

    try:
        from PIL import Image as PILImage
        import io

        def make_image_bytes(color):
            img = PILImage.new("RGB", (224, 224), color=color)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()

        # 3 synthetic images
        images = [
            ("img1.jpg", make_image_bytes((180, 120, 100)), "image/jpeg"),
            ("img2.jpg", make_image_bytes((120, 180, 140)), "image/jpeg"),
            ("img3.jpg", make_image_bytes((100, 100, 180)), "image/jpeg"),
        ]

        r = requests.post(
            f"{base_url}/predict/batch",
            files=[("files", img) for img in images],
            timeout=60
        )
        assert r.status_code == 200, f"Status {r.status_code}: {r.text}"
        data = r.json()

        log("PASS", f"Batch returns 200")
        log("PASS", f"Total images    : {data['total_images']}")
        log("PASS", f"Total time      : {data['total_time_ms']} ms")
        log("PASS" if len(data["results"]) == 3 else "FAIL",
            f"Results count   : {len(data['results'])}")

        for i, res in enumerate(data["results"]):
            log("PASS",
                f"Image {i+1}: {res['predicted_class']} "
                f"({res['confidence_pct']})")

    except ImportError:
        log("SKIP", "Pillow not installed — skipping batch test")
    except Exception as e:
        log("FAIL", "POST /predict/batch", str(e))


def test_batch_too_many(base_url):
    separator("POST /predict/batch — Limit Validation")
    try:
        from PIL import Image as PILImage
        import io

        def make_bytes():
            img = PILImage.new("RGB", (50, 50), color=(100, 100, 100))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()

        # Send 11 images (limit is 10)
        files = [("files", (f"img{i}.jpg", make_bytes(), "image/jpeg")) for i in range(11)]
        r = requests.post(f"{base_url}/predict/batch", files=files, timeout=30)
        assert r.status_code == 400
        log("PASS", "11 images → 400 Bad Request (limit enforced)")
    except ImportError:
        log("SKIP", "Pillow not installed")
    except Exception as e:
        log("FAIL", "Batch limit validation", str(e))


# ── Main 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the HAM10000 FastAPI")
    parser.add_argument("--url",   default=DEFAULT_URL, help="API base URL")
    parser.add_argument("--image", default=None,        help="Path to a real image to test")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print("\n" + "═" * 55)
    print(f"  HAM10000 API TEST SUITE")
    print(f"  Target: {base_url}")
    print("═" * 55)

    # Check server is reachable
    try:
        requests.get(f"{base_url}/", timeout=5)
    except Exception:
        print(f"\n Server not reachable at {base_url}")
        print("   Start with:  uvicorn api.main:app --reload")
        print("   Or:          docker compose up\n")
        sys.exit(1)

    test_root(base_url)
    test_health(base_url)
    test_classes(base_url)
    test_predict_no_file(base_url)
    test_predict_with_image(base_url, args.image)
    test_batch_predict(base_url, args.image)
    test_batch_too_many(base_url)

    # ── Summary 
    separator("SUMMARY")
    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] == "FAIL")
    skipped = sum(1 for r in results if r[0] == "SKIP")
    total  = len(results)

    print(f"  Total  : {total}")
    print(f"   Pass : {passed}")
    print(f"   Fail : {failed}")
    print(f"   Skip : {skipped}")
    separator()

    sys.exit(0 if failed == 0 else 1)
