from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx

app = Flask(__name__)

def extract_text(file):
    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.filename.endswith(".docx"):
        doc = docx.Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        return file.read().decode("utf-8")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jd_file = request.files["jd"]
        resumes = request.files.getlist("resumes")
        top_n = int(request.form.get("top_n", 3))  # default to 3

        jd_text = extract_text(jd_file)
        resume_texts = [extract_text(r) for r in resumes]

        vectorizer = TfidfVectorizer().fit([jd_text] + resume_texts)
        jd_vec = vectorizer.transform([jd_text])
        resume_vecs = vectorizer.transform(resume_texts)

        scores = cosine_similarity(jd_vec, resume_vecs)[0]
        ranked = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)

        top_matches = [(r.filename, round(s * 100, 2)) for r, s in ranked[:top_n]]
        return render_template("index.html", results=top_matches)

    return render_template("index.html", results=None)

if __name__ == "__main__":
    app.run(debug=True)
