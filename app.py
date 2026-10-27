# GAN Image Quality and Diversity Analyzer
# Idea: we don't train a GAN. We either load a pretrained one (DCGAN from
# PyTorch GAN Zoo) or upload a folder of already generated images, and then
# check how diverse / repetitive the samples are.

import io
import zipfile

import numpy as np
import streamlit as st
import torch
from PIL import Image

st.set_page_config(page_title="GAN Diversity Analyzer", layout="wide")
st.title("GAN Image Quality and Diversity Analyzer")
st.write(
    "Load images from a pretrained GAN or upload your own generated images. "
    "The app checks how many are unique and whether the set looks diverse or repetitive."
)

IMG_SIZE = 64  # every image is resized to this before comparing


# ---------------- loading images ----------------

@st.cache_resource
def load_gan():
    # pretrained DCGAN (trained on FashionGen), downloads the first time
    model = torch.hub.load(
        "facebookresearch/pytorch_GAN_zoo:hub", "DCGAN",
        pretrained=True, useGPU=False
    )
    return model


def generate_from_gan(n, seed):
    model = load_gan()
    torch.manual_seed(seed)
    noise, _ = model.buildNoiseData(n)
    with torch.no_grad():
        out = model.test(noise)  # shape: n x 3 x 64 x 64, values around -1..1

    images = []
    for img in out:
        img = img.clamp(-1, 1)
        img = ((img + 1) / 2 * 255).byte()
        img = img.permute(1, 2, 0).numpy()
        images.append(Image.fromarray(img))
    return images


def make_demo_images(n, seed):
    # offline demo: simple random shapes, so the app works without internet
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(n):
        bg = tuple(int(c) for c in rng.integers(0, 256, 3))
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), bg)
        px = np.array(img)
        for _ in range(rng.integers(1, 4)):
            color = rng.integers(0, 256, 3)
            x1, y1 = rng.integers(0, 40, 2)
            w, h = rng.integers(10, 30, 2)
            if rng.random() < 0.5:
                px[y1:y1 + h, x1:x1 + w] = color  # rectangle
            else:
                yy, xx = np.ogrid[:IMG_SIZE, :IMG_SIZE]
                mask = (xx - x1 - w // 2) ** 2 + (yy - y1 - w // 2) ** 2 <= (w // 2) ** 2
                px[mask] = color  # circle
        images.append(Image.fromarray(px))
    return images


def read_uploaded(files):
    images = []
    for f in files:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for name in z.namelist():
                    if name.lower().endswith((".png", ".jpg", ".jpeg")):
                        images.append(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
        else:
            images.append(Image.open(f).convert("RGB"))
    return images


# ---------------- analysis helpers ----------------

def average_hash(img, hash_size=8):
    # shrink to 8x8 grey, then each pixel -> 1 if brighter than average
    small = img.convert("L").resize((hash_size, hash_size))
    pixels = np.array(small, dtype=np.float32)
    return (pixels > pixels.mean()).flatten()


def count_unique(images, max_hamming):
    # two images are treated as the same if their hashes differ in very few bits
    hashes = [average_hash(img) for img in images]
    groups = []  # each group holds indexes of near-identical images
    for i, h in enumerate(hashes):
        placed = False
        for g in groups:
            if np.sum(h != hashes[g[0]]) <= max_hamming:
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return groups


def image_features(images):
    # simple feature vector = resized pixel values, centred and normalised
    feats = []
    for img in images:
        arr = np.array(img.resize((32, 32)), dtype=np.float32).flatten() / 255.0
        arr = arr - arr.mean()
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        feats.append(arr)
    return np.array(feats)


def similarity_matrix(feats):
    # cosine similarity between every pair of images
    return feats @ feats.T


def pca_2d(feats):
    # project features to 2D using SVD, to see how spread out the samples are
    centred = feats - feats.mean(axis=0)
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


def heatmap_image(sim, cell=12):
    # turn similarity values (-1..1) into a coloured image: blue = different, red = similar
    vals = (sim + 1) / 2
    r = (vals * 255).astype(np.uint8)
    b = ((1 - vals) * 255).astype(np.uint8)
    g = np.full_like(r, 60)
    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb)
    size = sim.shape[0] * cell
    return img.resize((size, size), Image.NEAREST)


# ---------------- sidebar ----------------

st.sidebar.header("Settings")
source = st.sidebar.radio(
    "Image source",
    ["Demo images (offline)", "Pretrained GAN (DCGAN)", "Upload generated images"]
)

images = []
if source == "Demo images (offline)":
    n_demo = st.sidebar.slider("Number of images", 8, 64, 24, step=4)
    demo_seed = st.sidebar.number_input("Random seed", value=7, step=1)
    st.session_state["images"] = make_demo_images(n_demo, int(demo_seed))
elif source == "Pretrained GAN (DCGAN)":
    n_images = st.sidebar.slider("Number of images", 8, 64, 24, step=4)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)
    if st.sidebar.button("Generate"):
        with st.spinner("Loading GAN and generating images..."):
            try:
                st.session_state["images"] = generate_from_gan(n_images, int(seed))
            except Exception as e:
                st.error(f"Could not load the GAN (needs internet the first time): {e}")
else:
    files = st.sidebar.file_uploader(
        "Upload images or a .zip", type=["png", "jpg", "jpeg", "zip"],
        accept_multiple_files=True
    )
    if files:
        st.session_state["images"] = read_uploaded(files)

# for demo: copy a few images to show what mode collapse looks like
simulate = st.sidebar.checkbox("Simulate mode collapse (repeat some images)")
max_hamming = st.sidebar.slider("Duplicate threshold (hash bit difference)", 0, 20, 5)
sim_threshold = st.sidebar.slider("High similarity threshold", 0.5, 1.0, 0.9, step=0.01)

images = list(st.session_state.get("images", []))
if simulate and len(images) >= 4:
    images = images[: len(images) // 2] + images[: len(images) // 2]

if len(images) < 2:
    st.info("Generate or upload at least 2 images to start the analysis.")
    st.stop()

images = [img.resize((IMG_SIZE, IMG_SIZE)) for img in images]


# ---------------- 1. gallery ----------------

st.subheader("1. Generated Image Gallery")
cols = st.columns(8)
for i, img in enumerate(images):
    cols[i % 8].image(img, caption=f"#{i}", width=100)


# ---------------- 2. unique images + similarity ----------------

st.subheader("2. Uniqueness and Similarity")
groups = count_unique(images, max_hamming)
n_total = len(images)
n_unique = len(groups)
unique_ratio = n_unique / n_total

feats = image_features(images)
sim = similarity_matrix(feats)
upper = sim[np.triu_indices(n_total, k=1)]  # each pair only once
avg_sim = float(upper.mean())
high_pairs = int(np.sum(upper >= sim_threshold))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total images", n_total)
c2.metric("Unique images", n_unique)
c3.metric("Average similarity", f"{avg_sim:.3f}")
c4.metric("Very similar pairs", high_pairs)

dup_groups = [g for g in groups if len(g) > 1]
if dup_groups:
    st.write("Near-duplicate groups found:")
    for g in dup_groups:
        st.write("Images " + ", ".join(f"#{i}" for i in g))
else:
    st.write("No near-duplicate images found.")


# ---------------- 3. diversity visualisation ----------------

st.subheader("3. Diversity Visualization")
left, right = st.columns(2)

with left:
    st.write("**Pairwise similarity heatmap** (red = similar, blue = different)")
    st.image(heatmap_image(sim))

with right:
    st.write("**2D spread of samples (PCA)** – points far apart means more diversity")
    pts = pca_2d(feats)
    st.scatter_chart({"x": pts[:, 0], "y": pts[:, 1]}, x="x", y="y")

st.write("**Distribution of pairwise similarity**")
counts, edges = np.histogram(upper, bins=20, range=(-1, 1))
labels = [f"{edges[i]:.1f}" for i in range(len(counts))]
st.bar_chart({"similarity": labels, "pairs": counts}, x="similarity", y="pairs")


# ---------------- 4. report ----------------

st.subheader("4. Diversity Report")
diversity_score = (1 - avg_sim) * unique_ratio  # higher = more diverse

if unique_ratio >= 0.9 and avg_sim < 0.6:
    verdict = "DIVERSE"
    st.success("The sample set appears DIVERSE.")
elif unique_ratio < 0.7 or avg_sim >= 0.8:
    verdict = "REPETITIVE"
    st.error("The sample set appears REPETITIVE (possible mode collapse).")
else:
    verdict = "MODERATELY DIVERSE"
    st.warning("The sample set is MODERATELY DIVERSE, some images look alike.")

report = f"""
- Total images analysed: **{n_total}**
- Unique images (by perceptual hash): **{n_unique}** ({unique_ratio * 100:.1f}%)
- Average pairwise cosine similarity: **{avg_sim:.3f}**
- Pairs above similarity {sim_threshold}: **{high_pairs}**
- Diversity score (0 to 2, higher is better): **{diversity_score:.3f}**
- Verdict: **{verdict}**
"""
st.markdown(report)

st.download_button("Download report", report, file_name="diversity_report.md")
