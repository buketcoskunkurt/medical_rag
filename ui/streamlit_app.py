import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Medical Assistant", page_icon="🩺", layout="centered")
st.title("Medical Assistant 🩺")
st.caption("Ask a medical question; the app will retrieve evidence and generate a one-sentence answer.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL, help="Base URL of the FastAPI service")
    k = st.number_input("Top-k snippets", min_value=1, max_value=20, value=5, step=1)

q = st.text_input("Your question", value="", placeholder="Type your medical question…")

run = st.button("Ask", type="primary")

def ask(api_base: str, question: str, k: int):
    try:
        r = requests.post(f"{api_base}/qa", json={"question": question, "k": int(k)}, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def looks_turkish(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    # Heuristic: Turkish-specific characters or common Turkish words
    if any(ch in t for ch in "çğıöşüÇĞİÖŞÜ"):  # accented letters
        return True
    lw = t.lower()
    common = [" ve ", " ile ", " nedir", " nasıl", " hangi", " ne ", " mıdır", " mı ", " mi ", " nelerdir"]
    return any(w in lw for w in common)

if run and q.strip():
    with st.spinner("Contacting API…"):
        data = ask(api_url.strip().rstrip('/'), q.strip(), k)
    if data:
        eng = ((data.get("english") or {}).get("text") or "").strip()
        tr = ((data.get("turkish") or {}).get("text") or "").strip()
        # If user asked in Turkish, prefer Turkish answer; else English
        ans = tr if looks_turkish(q) and tr else eng or tr
        st.subheader("Answer")
        st.write(ans)

        st.subheader("References")
        used = data.get("used_snippets") or []
        if not used:
            st.caption("No snippets available.")
        else:
            for i, s in enumerate(used, 1):
                title = s.get("title") or s.get("title_en") or "(no title)"
                url = s.get("url") or ""
                src = s.get("source") or ""
                line = f"{i}. {title}"
                if url:
                    line += f" — [link]({url})"
                if src:
                    line += f" • source: {src}"
                st.markdown(line)

        st.caption(
            f"Retrieval: {data.get('retrieval_time_seconds', 0)}s • Generation: {data.get('generation_time_seconds', 0)}s • Total: {data.get('total_time_seconds', 0)}s"
        )
