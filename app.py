import os, io, json, base64, uuid
from datetime import datetime
from functools import wraps
import cv2
import numpy as np
from PIL import Image
import torch
import timm
import segmentation_models_pytorch as smp
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template_string, jsonify, send_file, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from fpdf import FPDF

PROJECT_DIR    = "."
UNET_CKPT_PATH = f"{PROJECT_DIR}/models/unet_lesion.pth"
CLF_CKPT_PATH  = f"{PROJECT_DIR}/models/efficientnet_b3.pth"
FAISS_IDX_PATH = f"{PROJECT_DIR}/rag_index/derma_faiss.index"
RAG_DOCS_PATH  = f"{PROJECT_DIR}/rag_index/derma_docs.npy"
USERS_DB_PATH  = f"{PROJECT_DIR}/data/users_db.json"
CASES_DB_PATH  = f"{PROJECT_DIR}/data/cases_db.json"

os.makedirs(f"{PROJECT_DIR}/data", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet_ckpt = torch.load(UNET_CKPT_PATH, map_location=device)
unet_model = smp.Unet(encoder_name=unet_ckpt["encoder"], encoder_weights=None, in_channels=3, classes=1).to(device)
unet_model.load_state_dict({k: v.float() for k, v in unet_ckpt["state_dict"].items()})
unet_model.eval()

clf_ckpt = torch.load(CLF_CKPT_PATH, map_location=device)
clf_model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=7).to(device)
clf_model.load_state_dict({k: v.float() for k, v in clf_ckpt["state_dict"].items()})
clf_model.eval()
idx_to_diag = clf_ckpt["idx_to_diag"]

cam = GradCAM(model=clf_model, target_layers=[clf_model.conv_head])
rag_docs  = np.load(RAG_DOCS_PATH, allow_pickle=True).tolist()
tfidf_vec = TfidfVectorizer(stop_words="english")
tfidf_mat = tfidf_vec.fit_transform(rag_docs)
def rag_search(query, k=2):
    sims = cosine_similarity(tfidf_vec.transform([query]), tfidf_mat)[0]
    top_idx = np.argsort(sims)[::-1][:k]
    return [rag_docs[i] for i in top_idx]

MALIGNANT_SPECTRUM = {"Melanoma", "Basal Cell Carcinoma", "Actinic Keratosis"}

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f: return json.load(f)
        except Exception: return default
    return default

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

USERS_DB = load_json(USERS_DB_PATH, {})
CASES_DB = load_json(CASES_DB_PATH, [])

def img_to_b64(rgb_arr):
    buf = io.BytesIO()
    Image.fromarray(rgb_arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def get_risk_tier(malig_prob):
    if malig_prob >= 50.0:
        return {"tier": "High Malignancy Probability", "level": "high", "advice": "Prompt dermatoscopic biopsy & specialist consultation recommended."}
    elif malig_prob >= 20.0:
        return {"tier": "Moderate / Indeterminate Probability", "level": "moderate", "advice": "Clinical dermoscopy review & short-term digital follow-up advised."}
    else:
        return {"tier": "Low Malignancy Probability", "level": "low", "advice": "Routine annual skin exam & monthly ABCDE self-monitoring."}

def generate_local_clinical_report(patient_id, lesion_site, pred_label, pred_prob, malig_prob, benign_prob, area_pct, risk_info, top_diffs):
    retrieved_guideline = rag_search(f"{pred_label} clinical management wellness", 2)[0]
    wellness_guideline  = rag_docs[-1]
    diff_str = ", ".join([f"{d['label']} ({d['prob']}%)" for d in top_diffs[:4]])
    return (
        f"AI PROBABILITY ASSESSMENT: {malig_prob}% Malignant / Dysplastic Probability  |  {benign_prob}% Benign Probability\n"
        f"Risk Tier: {risk_info['tier']} — Primary Pattern: {pred_label} ({pred_prob}%)\n"
        f"-------------------------------------------------------------------------\n"
        f"1. CLINICAL PROBABILITY & DIFFERENTIAL SUMMARY\n"
        f"• Patient / Lesion ID: {patient_id} | Anatomical Site: {lesion_site}\n"
        f"• Top Differential Probabilities: {diff_str}\n"
        f"• Segmented Lesion Coverage (U-Net): {area_pct}% of dermoscopic frame\n"
        f"• Clinical Reference (FAISS RAG): {retrieved_guideline}\n"
        f"• Recommended Next Step: {risk_info['advice']}\n\n"
        f"2. PATIENT WELLNESS & MONITORING GUIDANCE\n"
        f"• Understanding Your Probability Score: This AI screening model estimates a {malig_prob}% probability of malignant/precancerous features and a {benign_prob}% probability of benign features. Because AI probabilities are statistical estimates rather than a definitive diagnosis, an in-person dermatologist check is always recommended.\n"
        f"• Preventive Protocol: {wellness_guideline}"
    )

def generate_local_chat_reply(user_msg):
    q_lower = user_msg.lower()
    best_doc, second_doc = rag_search(user_msg, 2)
    if CASES_DB and any(k in q_lower for k in ["latest", "scan", "case", "result", "prob", "percent", "risk", "patient", "score"]):
        c = CASES_DB[-1]
        return (
            f"Probability Breakdown for {c['patient_id']} ({c['lesion_site']}):\n"
            f"• Malignant / Dysplastic Probability: {c['malig_prob']}%\n"
            f"• Benign Probability: {c['benign_prob']}%\n"
            f"• Top Class Match: {c['pred_label']} ({c['confidence']}%)\n"
            f"• Assessment Tier: {c['risk_tier']}\n"
            f"• Guidance: {c['risk_advice']}\n\n"
            f"Relevant Clinical Reference:\n{best_doc}"
        )
    return f"DermaGPT Clinical Reference:\n\n• {best_doc}\n\n• {second_doc}"

app = Flask(__name__)
app.secret_key = "dermagpt_hf_permanent_portal_secret"

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_email" not in session: return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

AUTH_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DermaGPT AI | Clinical Access Portal</title><script src="https://cdn.tailwindcss.com"></script></head>
<body class="min-h-screen bg-[#0b111a] text-slate-200 flex items-center justify-center p-4">
  <div class="max-w-4xl w-full bg-[#131c29] border border-slate-800 rounded-3xl shadow-2xl overflow-hidden grid grid-cols-1 md:grid-cols-12">
    <div class="md:col-span-5 bg-gradient-to-b from-[#162738] to-[#0e1622] p-8 flex flex-col justify-between border-r border-slate-800/80">
      <div>
        <div class="flex items-center gap-3 mb-8">
          <div class="w-10 h-10 rounded-xl bg-cyan-500/15 border border-cyan-400/30 flex items-center justify-center text-cyan-400 font-bold text-xl">+</div>
          <span class="text-2xl font-bold text-white">DermaGPT <span class="text-cyan-400">AI</span></span>
        </div>
        <h2 class="text-xl font-semibold text-white mb-3">Probabilistic Skin Lesion Analysis & Explainable AI</h2>
        <p class="text-xs text-slate-400 leading-relaxed">Sign in or create an account to evaluate dermoscopic probability distributions, U-Net lesion segmentation, and Grad-CAM heatmaps.</p>
      </div>
      <div class="pt-6 border-t border-slate-800 text-xs text-slate-400 space-y-1">
        <p class="text-emerald-400 font-medium">• 24/7 Cloud Clinical Engine Active</p>
        <p>Default Login: <span class="text-slate-200">clinician@dermagpt.ai</span> / <span class="text-slate-200">derma2026</span></p>
      </div>
    </div>
    <div class="md:col-span-7 p-8 md:p-10 flex flex-col justify-center">
      <div class="flex bg-[#0b111a] p-1 rounded-xl border border-slate-800 mb-6">
        <button type="button" onclick="switchTab('login')" class="flex-1 py-2 text-xs font-semibold rounded-lg bg-[#1d2b3d] text-cyan-300 cursor-pointer">Sign In</button>
        <button type="button" onclick="switchTab('register')" class="flex-1 py-2 text-xs font-semibold rounded-lg text-slate-400 hover:text-white cursor-pointer">Create New Account</button>
      </div>
      {% if error %}<div class="mb-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs">{{ error }}</div>{% endif %}
      <form id="form-login" method="POST" action="/login" class="space-y-4">
        <h3 class="text-lg font-bold text-white">Sign In</h3>
        <input type="email" name="email" required value="clinician@dermagpt.ai" class="w-full px-4 py-2.5 rounded-xl bg-[#0b111a] border border-slate-700 text-white text-sm outline-none" />
        <input type="password" name="password" required value="derma2026" class="w-full px-4 py-2.5 rounded-xl bg-[#0b111a] border border-slate-700 text-white text-sm outline-none" />
        <button type="submit" class="w-full py-3 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-sm cursor-pointer">Access Clinical Dashboard</button>
      </form>
      <form id="form-register" method="POST" action="/register" class="hidden space-y-3.5">
        <h3 class="text-lg font-bold text-white">Create User Account</h3>
        <input type="text" name="full_name" required placeholder="Full Name" class="w-full px-4 py-2.5 rounded-xl bg-[#0b111a] border border-slate-700 text-white text-sm outline-none" />
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <input type="email" name="email" required placeholder="Email Address" class="w-full px-4 py-2.5 rounded-xl bg-[#0b111a] border border-slate-700 text-white text-sm outline-none" />
          <select name="role" class="w-full px-3.5 py-2.5 rounded-xl bg-[#0b111a] border border-slate-700 text-white text-sm outline-none">
            <option value="Dermatologist">Dermatologist</option>
            <option value="General Physician">General Physician</option>
            <option value="Medical Researcher">Medical Researcher</option>
            <option value="Patient / Wellness User">Patient / Wellness User</option>
          </select>
        </div>
        <input type="password" name="password" required minlength="4" placeholder="Create Password" class="w-full px-4 py-2.5 rounded-xl bg-[#0b111a] border border-slate-700 text-white text-sm outline-none" />
        <button type="submit" class="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-sm cursor-pointer">Register & Launch Dashboard</button>
      </form>
    </div>
  </div>
  <script>
    function switchTab(mode) {
      document.getElementById('form-login').classList.toggle('hidden', mode === 'register');
      document.getElementById('form-register').classList.toggle('hidden', mode !== 'register');
    }
  </script>
</body></html>"""

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DermaGPT AI | Probabilistic Clinical Dashboard</title><script src="https://cdn.tailwindcss.com"></script><script src="https://cdn.jsdelivr.net/npm/chart.js"></script></head>
<body class="min-h-screen bg-[#0b111a] text-slate-200 flex overflow-x-hidden">
  <aside class="w-64 bg-[#0f1722] border-r border-slate-800/80 hidden lg:flex flex-col justify-between shrink-0 sticky top-0 h-screen">
    <div>
      <div class="h-16 px-6 flex items-center gap-3 border-b border-slate-800/80">
        <div class="w-8 h-8 rounded-lg bg-cyan-500/15 border border-cyan-400/30 flex items-center justify-center text-cyan-400 font-bold text-lg">+</div>
        <span class="text-lg font-bold text-white">DermaGPT <span class="text-cyan-400">AI</span></span>
      </div>
      <nav class="p-4 space-y-1.5 text-sm">
        <a href="#section-upload" class="block px-3.5 py-2.5 rounded-xl bg-cyan-500/15 text-cyan-300 border border-cyan-500/30 font-medium">Dashboard</a>
        <a href="#section-queue" class="flex justify-between px-3.5 py-2.5 rounded-xl text-slate-400 hover:bg-slate-800/60 hover:text-white"><span>Patient Queue</span><span class="px-2 py-0.5 text-[11px] rounded-full bg-slate-800">{{ cases|length }}</span></a>
        <a href="#section-xai" class="block px-3.5 py-2.5 rounded-xl text-slate-400 hover:bg-slate-800/60 hover:text-white">Dermoscopy Scans</a>
        <a href="#section-analytics" class="block px-3.5 py-2.5 rounded-xl text-slate-400 hover:bg-slate-800/60 hover:text-white">Probability Analytics</a>
      </nav>
    </div>
    <div class="p-4 border-t border-slate-800/80 space-y-3">
      <div class="px-2"><p class="text-xs font-semibold text-white truncate">{{ user_name }}</p><p class="text-[11px] text-slate-400">{{ user_role }}</p></div>
      <a href="/logout" class="block text-center w-full py-2 rounded-xl bg-slate-800 hover:bg-rose-500/20 hover:text-rose-300 text-xs font-semibold">Sign Out</a>
    </div>
  </aside>

  <div class="flex-1 flex flex-col min-w-0">
    <header class="h-16 bg-[#0f1722]/90 border-b border-slate-800/80 px-6 flex items-center justify-between sticky top-0 z-20 gap-4">
      <input type="text" id="global-search" onkeyup="filterQueue()" placeholder="Search patient ID, class, probability..." class="w-full max-w-md px-4 py-2 rounded-xl bg-[#16202e] border border-slate-700/70 text-xs text-slate-200 outline-none" />
      <span class="px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-medium">Cloud Engine Active</span>
    </header>

    <main class="p-6 space-y-6 max-w-[1440px] w-full mx-auto">
      {% if result %}
      <div class="p-5 rounded-2xl border shadow-lg {% if result.risk_level == 'high' %}bg-rose-950/80 border-rose-500/50{% elif result.risk_level == 'moderate' %}bg-amber-950/80 border-amber-500/50{% else %}bg-cyan-950/80 border-cyan-500/40{% endif %}">
        <div class="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <span class="px-2.5 py-0.5 rounded-md text-[11px] font-bold uppercase bg-cyan-500 text-slate-950">{{ result.risk_tier }}</span>
            <h2 class="text-lg sm:text-xl font-bold text-white mt-1.5">Malignancy Probability: <span class="text-rose-400">{{ result.malig_prob }}%</span>  |  Benign Probability: <span class="text-emerald-400">{{ result.benign_prob }}%</span></h2>
            <p class="text-xs text-slate-300 mt-1">Top Class Match: <strong class="text-white">{{ result.pred_label }} ({{ result.confidence }}%)</strong> • {{ result.risk_advice }}</p>
          </div>
          <a href="/download_report/{{ result.case_id }}" class="px-4 py-2.5 rounded-xl bg-white/10 hover:bg-white/20 text-white text-xs font-semibold">Download Clinical PDF</a>
        </div>
        <div class="w-full h-3.5 rounded-full bg-slate-800 overflow-hidden flex">
          <div class="bg-rose-500 h-full" style="width: {{ result.malig_prob }}%"></div>
          <div class="bg-emerald-400 h-full" style="width: {{ result.benign_prob }}%"></div>
        </div>
      </div>
      {% endif %}

      <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <div class="bg-[#131c29] border border-slate-800 p-5 rounded-2xl"><p class="text-xs text-slate-400">Elevated Risk Scans (≥20%)</p><p class="text-2xl font-bold text-rose-400 mt-1">{{ stats.elevated_count }} Cases</p></div>
        <div class="bg-[#131c29] border border-slate-800 p-5 rounded-2xl"><p class="text-xs text-slate-400">Total Scans Processed</p><p class="text-2xl font-bold text-white mt-1">{{ stats.total }}</p></div>
        <div class="bg-[#131c29] border border-slate-800 p-5 rounded-2xl"><p class="text-xs text-slate-400">Avg. Malignancy Probability</p><p class="text-2xl font-bold text-cyan-400 mt-1">{{ stats.avg_malig }}%</p></div>
        <div class="bg-[#131c29] border border-slate-800 p-5 rounded-2xl"><p class="text-xs text-slate-400">Avg. Segmented Area</p><p class="text-2xl font-bold text-white mt-1">{{ stats.avg_area }}%</p></div>
      </div>

      <div id="section-upload" class="bg-[#131c29] border border-slate-800 p-5 rounded-2xl">
        <form method="POST" enctype="multipart/form-data" class="grid grid-cols-1 md:grid-cols-12 gap-4 items-end">
          <div class="md:col-span-3"><label class="block text-[11px] uppercase text-slate-400 mb-1">Patient / Lesion ID</label><input type="text" name="patient_id" placeholder="e.g. PT-40294" required class="w-full px-3.5 py-2.5 rounded-xl bg-[#0b111a] border border-slate-700 text-sm text-white outline-none" /></div>
          <div class="md:col-span-3"><label class="block text-[11px] uppercase text-slate-400 mb-1">Anatomical Site</label><input type="text" name="lesion_site" value="Upper Back" class="w-full px-3.5 py-2.5 rounded-xl bg-[#0b111a] border border-slate-700 text-sm text-white outline-none" /></div>
          <div class="md:col-span-4"><label class="block text-[11px] uppercase text-slate-400 mb-1">Dermoscopy Image</label><input type="file" name="lesion_image" accept="image/*" required class="w-full text-xs text-slate-400 bg-[#0b111a] border border-slate-700 rounded-xl p-1.5" /></div>
          <div class="md:col-span-2"><button type="submit" class="w-full py-2.5 px-4 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-sm cursor-pointer">Analyze Scan</button></div>
        </form>
      </div>

      {% if result %}
      <div id="section-xai" class="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div class="bg-[#131c29] border border-slate-800 p-4 rounded-2xl"><h3 class="text-sm font-semibold text-white mb-2">Original Dermoscopy</h3><img src="data:image/png;base64,{{ result.orig_b64 }}" class="w-full h-56 object-cover rounded-xl" /></div>
        <div class="bg-[#131c29] border border-slate-800 p-4 rounded-2xl"><h3 class="text-sm font-semibold text-white mb-2">U-Net Segmentation ({{ result.area_pct }}%)</h3><img src="data:image/png;base64,{{ result.mask_overlay_b64 }}" class="w-full h-56 object-cover rounded-xl" /></div>
        <div class="bg-[#131c29] border border-slate-800 p-4 rounded-2xl"><h3 class="text-sm font-semibold text-white mb-2">Grad-CAM XAI Heatmap</h3><img src="data:image/png;base64,{{ result.cam_b64 }}" class="w-full h-56 object-cover rounded-xl" /></div>
      </div>
      {% endif %}

      <div class="grid grid-cols-1 xl:grid-cols-12 gap-5">
        <div id="section-analytics" class="xl:col-span-5 bg-[#131c29] border border-slate-800 p-5 rounded-2xl">
          <h3 class="text-sm font-bold text-white mb-4">7-Class Probability Distribution (%)</h3>
          {% if result %}
          <div class="space-y-3">
            {% for item in result.prob_bars %}
            <div>
              <div class="flex justify-between text-xs mb-1"><span class="text-slate-300 font-medium">{{ item.label }}</span><span class="font-bold {{ 'text-rose-400' if item.is_cancer else 'text-cyan-300' }}">{{ item.prob }}%</span></div>
              <div class="w-full h-2 rounded-full bg-[#0b111a]"><div class="h-full rounded-full {{ 'bg-rose-500' if item.is_cancer else 'bg-cyan-500' }}" style="width: {{ [item.prob, 2]|max }}%"></div></div>
            </div>
            {% endfor %}
          </div>
          {% else %}
          <div class="h-52"><canvas id="portfolioChart"></canvas></div>
          {% endif %}
        </div>

        <div id="section-queue" class="xl:col-span-7 bg-[#131c29] border border-slate-800 p-5 rounded-2xl overflow-x-auto">
          <h3 class="text-sm font-bold text-white mb-4">Patient Probability Queue ({{ cases|length }})</h3>
          <table class="w-full text-left text-xs">
            <thead><tr class="border-b border-slate-800 text-slate-400 uppercase text-[11px]"><th class="py-2 px-3">Patient</th><th class="py-2 px-3">Malignant / Benign Prob.</th><th class="py-2 px-3">Date</th><th class="py-2 px-3 text-right">Actions</th></tr></thead>
            <tbody class="divide-y divide-slate-800/60">
              {% for c in cases %}
              <tr class="queue-row hover:bg-slate-800/30">
                <td class="py-3 px-3"><p class="font-bold text-white">{{ c.patient_id }}</p><p class="text-slate-400">{{ c.pred_label }} ({{ c.confidence }}%)</p></td>
                <td class="py-3 px-3"><span class="font-bold text-rose-400">{{ c.malig_prob }}% Malig.</span> / <span class="font-semibold text-emerald-400">{{ c.benign_prob }}% Benign</span></td>
                <td class="py-3 px-3 text-slate-400">{{ c.timestamp.split(' ')[0] }}</td>
                <td class="py-3 px-3 text-right space-x-1">
                  <a href="/download_report/{{ c.case_id }}" class="px-2.5 py-1.5 rounded-lg bg-[#1b2738] hover:bg-cyan-600 text-white font-semibold">PDF</a>
                  <form method="POST" action="/delete_case/{{ c.case_id }}" class="inline" onsubmit="return confirm('Delete case?');"><button type="submit" class="px-2.5 py-1.5 rounded-lg bg-[#1b2738] hover:bg-rose-600 text-white font-semibold cursor-pointer">Delete</button></form>
                </td>
              </tr>
              {% else %}
              <tr><td colspan="4" class="py-8 text-center text-slate-500">No scans in queue yet. Upload an image above.</td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>

      {% if result %}
      <div class="bg-[#131c29] border border-slate-800 p-6 rounded-2xl">
        <h3 class="text-sm font-bold text-white mb-3">Clinical Probability & Wellness Synthesis Report</h3>
        <div class="p-4 rounded-xl bg-[#0b111a] border border-slate-800 text-xs sm:text-sm text-slate-300 whitespace-pre-wrap">{{ result.report }}</div>
      </div>
      {% endif %}
    </main>
  </div>

  <!-- LARGE MEDICAL BOT ICON & CLICK-TO-OPEN CHATBOT -->
  <div class="fixed bottom-6 right-6 z-50 flex flex-col items-end">
    <div id="chatbot-window" class="hidden mb-4 w-80 sm:w-96 bg-[#131c29] rounded-3xl shadow-2xl border border-cyan-500/30 overflow-hidden flex flex-col">
      <div class="bg-gradient-to-r from-[#162b40] to-[#0f1e2e] px-4 py-3.5 flex justify-between items-center border-b border-slate-700/80">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 rounded-full bg-cyan-500/20 border border-cyan-400/50 flex items-center justify-center">
            <svg class="w-5 h-5 text-cyan-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="7" width="16" height="12" rx="4"/><path d="M12 3v4M9 12h.01M15 12h.01M9 16h6"/></svg>
          </div>
          <div><h4 class="text-xs font-bold text-white">DermaBot Medical Assistant</h4><p class="text-[11px] text-emerald-400">Online • RAG Clinical Support</p></div>
        </div>
        <button type="button" onclick="toggleChatbot()" class="text-slate-400 hover:text-white text-xl font-bold px-1.5 cursor-pointer">×</button>
      </div>
      <div id="chat-messages" class="p-4 h-72 overflow-y-auto space-y-3 bg-[#0b111a] text-xs">
        <div class="bg-[#16202e] border border-slate-800 p-3.5 rounded-2xl text-slate-200 leading-relaxed">Hello <strong>{{ user_name }}</strong>! Ask me about the latest malignancy probability score, ABCDE screening, or skin care.</div>
      </div>
      <div class="p-3 bg-[#131c29] border-t border-slate-800 flex gap-2">
        <input type="text" id="chat-input" placeholder="Ask DermaBot..." onkeydown="if(event.key==='Enter') sendChat()" class="flex-1 px-3.5 py-2.5 rounded-xl bg-[#0b111a] border border-slate-700 text-white text-xs outline-none" />
        <button type="button" onclick="sendChat()" class="px-4 py-2.5 rounded-xl bg-cyan-600 text-white text-xs font-semibold cursor-pointer">Send</button>
      </div>
    </div>
    <button type="button" onclick="toggleChatbot()" title="Open DermaBot Medical Assistant"
      class="relative w-16 h-16 rounded-full bg-gradient-to-br from-cyan-500 to-sky-700 hover:from-cyan-400 hover:to-sky-600 text-white shadow-2xl border-2 border-cyan-300/60 flex items-center justify-center transition transform hover:scale-110 cursor-pointer">
      <span class="absolute top-0.5 right-0.5 w-3.5 h-3.5 rounded-full bg-emerald-400 border-2 border-[#0b111a]"></span>
      <svg class="w-9 h-9 text-white" viewBox="0 0 64 64" fill="none">
        <line x1="32" y1="8" x2="32" y2="16" stroke="currentColor" stroke-width="3.5" stroke-linecap="round"/><circle cx="32" cy="7" r="3.5" fill="#67e8f9"/>
        <rect x="14" y="16" width="36" height="32" rx="10" fill="#0f172a" stroke="currentColor" stroke-width="3.2"/>
        <path d="M32 20v6M29 23h6" stroke="#34d399" stroke-width="2.8" stroke-linecap="round"/>
        <circle cx="24" cy="34" r="3.2" fill="#22d3ee"/><circle cx="40" cy="34" r="3.2" fill="#22d3ee"/>
        <path d="M25 41c2.5 2.5 11.5 2.5 14 0" stroke="currentColor" stroke-width="2.8" stroke-linecap="round"/>
      </svg>
    </button>
  </div>

  <script>
    {% if not result %}
    const chartData = {{ chart_payload | tojson }};
    new Chart(document.getElementById('portfolioChart'), {
      type: 'doughnut',
      data: { labels: chartData.dist_labels.length ? chartData.dist_labels : ['No Scans Yet'], datasets: [{ data: chartData.dist_counts.length ? chartData.dist_counts : [1], backgroundColor: chartData.dist_counts.length ? ['#f43f5e', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6'] : ['#1e293b'], borderWidth: 0 }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 11 } } } } }
    });
    {% endif %}
    function filterQueue() { const q = document.getElementById('global-search').value.toLowerCase(); document.querySelectorAll('.queue-row').forEach(r => r.style.display = r.innerText.toLowerCase().includes(q) ? '' : 'none'); }
    function toggleChatbot() { document.getElementById('chatbot-window').classList.toggle('hidden'); }
    async function sendChat() {
      const inp = document.getElementById('chat-input'), msg = inp.value.trim(); if (!msg) return;
      const box = document.getElementById('chat-messages');
      box.innerHTML += `<div class="text-right"><span class="inline-block bg-cyan-600 text-white px-3.5 py-2 rounded-2xl">${msg}</span></div>`; inp.value = '';
      const res = await fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({message: msg}) });
      const data = await res.json();
      box.innerHTML += `<div><div class="bg-[#16202e] p-3.5 rounded-2xl text-slate-200 whitespace-pre-wrap">${data.reply}</div></div>`; box.scrollTop = box.scrollHeight;
    }
  </script>
</body></html>"""

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        email, password = request.form.get("email", "").strip().lower(), request.form.get("password", "")
        user = USERS_DB.get(email)
        if user and check_password_hash(user["password_hash"], password):
            session["user_email"], session["user_name"], session["user_role"] = email, user["name"], user.get("role", "Clinician")
            return redirect(url_for("dashboard"))
        error = "Invalid credentials. Click Create New Account to register."
    return render_template_string(AUTH_HTML, error=error)

@app.route("/register", methods=["POST"])
def register():
    full_name, email, role, password = request.form.get("full_name", "").strip(), request.form.get("email", "").strip().lower(), request.form.get("role", "Clinician"), request.form.get("password", "")
    if email in USERS_DB: return render_template_string(AUTH_HTML, error="Email already registered. Please sign in.")
    USERS_DB[email] = {"name": full_name, "role": role, "password_hash": generate_password_hash(password)}
    save_json(USERS_DB_PATH, USERS_DB)
    session["user_email"], session["user_name"], session["user_role"] = email, full_name, role
    return redirect(url_for("dashboard"))

@app.route("/logout")
def logout():
    session.clear(); return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
@login_required
def dashboard():
    result = None
    if request.method == "POST":
        patient_id, lesion_site = request.form.get("patient_id", "PT-001").strip(), request.form.get("lesion_site", "Back").strip()
        orig_bgr = cv2.imdecode(np.frombuffer(request.files["lesion_image"].read(), np.uint8), cv2.IMREAD_COLOR)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = orig_rgb.shape
        mean, std = np.array([0.485, 0.456, 0.406], dtype=np.float32), np.array([0.229, 0.224, 0.225], dtype=np.float32)

        u_size = unet_ckpt["img_size"]
        u_tensor = torch.tensor((cv2.resize(orig_rgb, (u_size, u_size)).astype(np.float32)/255.0 - mean)/std).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad(): mask_bin = (torch.sigmoid(unet_model(u_tensor))[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        area_pct = round(float(mask_bin.mean()) * 100, 2)
        mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        overlay_rgb = orig_rgb.copy()
        teal_tint = np.zeros_like(orig_rgb); teal_tint[:, :] = (6, 182, 212)
        overlay_rgb[mask_resized == 1] = cv2.addWeighted(orig_rgb, 0.65, teal_tint, 0.35, 0)[mask_resized == 1]
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgb, contours, -1, (34, 211, 238), 2)

        c_size = clf_ckpt["img_size"]
        c_tensor = torch.tensor((cv2.resize(orig_rgb, (c_size, c_size)).astype(np.float32)/255.0 - mean)/std).permute(2,0,1).unsqueeze(0).to(device)
        with torch.no_grad(): probs = torch.softmax(clf_model(c_tensor), dim=1)[0].cpu().numpy()

        malig_prob  = round(sum(float(probs[i]) * 100 for i in range(7) if idx_to_diag[i] in MALIGNANT_SPECTRUM), 1)
        benign_prob = round(max(0.0, 100.0 - malig_prob), 1)
        pred_idx, pred_label, confidence = int(np.argmax(probs)), idx_to_diag[int(np.argmax(probs))], round(float(np.max(probs)) * 100, 1)
        risk_info = get_risk_tier(malig_prob)

        prob_bars = sorted([{"label": idx_to_diag[i], "prob": round(float(probs[i])*100, 1), "is_cancer": idx_to_diag[i] in MALIGNANT_SPECTRUM} for i in range(7)], key=lambda x: x["prob"], reverse=True)
        gradcam_rgb = show_cam_on_image(orig_rgb.astype(np.float32)/255.0, cv2.resize(cam(input_tensor=c_tensor, targets=[ClassifierOutputTarget(pred_idx)])[0], (w, h)), use_rgb=True)
        report_text = generate_local_clinical_report(patient_id, lesion_site, pred_label, confidence, malig_prob, benign_prob, area_pct, risk_info, prob_bars)

        case_record = {
            "case_id": uuid.uuid4().hex[:8], "patient_id": patient_id, "lesion_site": lesion_site, "clinician": session.get("user_name"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "pred_label": pred_label, "confidence": confidence,
            "malig_prob": malig_prob, "benign_prob": benign_prob, "area_pct": area_pct,
            "risk_tier": risk_info["tier"], "risk_level": risk_info["level"], "risk_advice": risk_info["advice"], "report": report_text
        }
        CASES_DB.append(case_record); save_json(CASES_DB_PATH, CASES_DB)
        result = {**case_record, "prob_bars": prob_bars, "orig_b64": img_to_b64(orig_rgb), "mask_overlay_b64": img_to_b64(overlay_rgb), "cam_b64": img_to_b64(gradcam_rgb)}

    total = len(CASES_DB)
    stats = {"total": total, "elevated_count": sum(1 for c in CASES_DB if c.get("malig_prob", 0) >= 20.0), "avg_malig": round(sum(c.get("malig_prob", 0) for c in CASES_DB)/total, 1) if total else 0.0, "avg_area": round(sum(c["area_pct"] for c in CASES_DB)/total, 1) if total else 0.0}
    dist_map = {}
    for c in CASES_DB: dist_map[c["pred_label"]] = dist_map.get(c["pred_label"], 0) + 1
    return render_template_string(DASHBOARD_HTML, result=result, cases=list(reversed(CASES_DB)), stats=stats, chart_payload={"dist_labels": list(dist_map.keys()), "dist_counts": list(dist_map.values())}, user_name=session.get("user_name"), user_role=session.get("user_role"))

@app.route("/delete_case/<case_id>", methods=["POST"])
@login_required
def delete_case(case_id):
    global CASES_DB
    CASES_DB = [c for c in CASES_DB if c["case_id"] != case_id]; save_json(CASES_DB_PATH, CASES_DB)
    return redirect(url_for("dashboard"))

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    return jsonify({"reply": generate_local_chat_reply(request.json.get("message", ""))})

@app.route("/download_report/<case_id>")
@login_required
def download_report(case_id):
    case = next((c for c in CASES_DB if c["case_id"] == case_id), None)
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"DermaGPT Probability Report - {case['patient_id']}", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, case["report"].encode("latin-1", "replace").decode("latin-1"))
    return send_file(io.BytesIO(pdf.output(dest="S").encode("latin-1")), as_attachment=True, download_name=f"DermaGPT_{case['patient_id']}.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
