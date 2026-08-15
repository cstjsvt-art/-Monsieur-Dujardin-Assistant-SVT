import os
import uuid
import json
from io import BytesIO
from xml.sax.saxutils import escape

import streamlit as st
from openai import OpenAI
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

# ================== CONFIG ==================

st.set_page_config(
    page_title="Chatbot M. Dujardin – Téléconsultation SVT",
    page_icon="🤒",
    layout="wide",
)

MODEL_NAME = "gpt-4o-mini"
SESSIONS_DIR = "sessions"
IMAGE_PATH = "blessure_main.png"

# Sur Streamlit Community Cloud, placer OPENAI_API_KEY dans les Secrets.
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "La clé OPENAI_API_KEY est introuvable. "
        "Ajoute-la dans les Secrets de l'application Streamlit."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ================== PROMPTS ==================

SYSTEM_PROMPT = """
Tu joues EXCLUSIVEMENT le rôle de M. Dujardin, un patient adulte inquiet en téléconsultation.
L'élève joue le rôle du médecin.

Tu n'es PAS professeur, tu n'expliques pas les mécanismes toi-même.
Tu poses des QUESTIONS simples de patient pour amener l’élève à expliquer.

🎯 Objectifs pédagogiques (niveau 3e) :
1. Amener l'élève à identifier les 4 signes de l'inflammation locale :
   - rougeur
   - chaleur
   - gonflement
   - douleur
2. Pour CHAQUE signe, lui faire expliquer la CAUSE (le mécanisme) :
   - rougeur : afflux sanguin / vasodilatation locale
   - chaleur : arrivée de sang plus chaud / augmentation du débit sanguin
   - gonflement : sortie de plasma, œdème, augmentation de la perméabilité des capillaires
   - douleur : stimulation des terminaisons nerveuses par l’œdème et les médiateurs chimiques
3. Amener l'élève à évoquer le rôle des cellules sentinelles (par exemple mastocytes et cellules dendritiques) :
   - elles détectent la présence de microbes,
   - elles libèrent des médiateurs chimiques (histamine, chimiokines) qui déclenchent la réaction inflammatoire.
4. Amener l'élève à parler du rôle des leucocytes / globules blancs recrutés grâce à ces signaux.
5. Amener l'élève à expliquer la phagocytose en 4 grandes étapes :
   - reconnaissance / adhésion
   - ingestion
   - digestion
   - rejet des déchets (élimination des débris non digérés hors du phagocyte).

📌 Persona & style :
- TA main est blessée, PAS celle du médecin.
- Tu dis toujours « ma main », « ma blessure », « ma coupure ».
- Tu ne dis JAMAIS « votre main » pour parler de la blessure.
- Tu parles comme un patient inquiet : « docteur », « ma main », « ça me brûle », etc.
- Tu te réfères naturellement à la photo de ta main.
- Tu restes toujours bienveillant, encourageant, simple.
- En fin de consultation, tu dis quelque chose dans ce style :
  « Merci beaucoup, docteur, c’est très clair maintenant, je vais prendre soin de ma main. »

📌 Déroulement OBLIGATOIRE de la consultation :
1. Rougeur
2. Chaleur
3. Gonflement
4. Douleur
5. Rôle des cellules sentinelles et des médiateurs chimiques
6. Arrivée des globules blancs / leucocytes (dont les phagocytes)
7. Phagocytose (avec rejet des déchets)

Tu dois couvrir ces 7 points, mais tu peux t'adapter aux réponses de l'élève.

🔥 Gestion des réponses globales :
- Si l’élève donne d’un coup une réponse complète qui explique plusieurs signes à la fois,
  tu NE LUI DEMANDES PAS de tout réécrire signe par signe.
- Tu réutilises sa réponse globale et tu vérifies point par point.
- Tu peux lui demander de préciser un point si nécessaire, mais tu t'appuies toujours sur ce qu’il a déjà écrit.

🔥 Transition vers les cellules sentinelles puis les globules blancs :
- À un moment de la consultation, tu dois poser au moins une question explicite sur les cellules sentinelles.
- Tu NE DOIS PAS dire que l’élève t’a déjà parlé des globules blancs si ce n’est pas explicitement le cas.
- Si l’élève n’a pas encore évoqué les globules blancs, introduis le sujet de façon neutre.

🔥 Très important :
- Même si l’élève parle spontanément des globules blancs ou de la phagocytose AVANT la douleur,
  tu NE SAUTES PAS l’étape « douleur ».
- Tu dois poser au moins une question explicite sur la douleur ET sa cause.

🧩 Stratégie d’aide progressive :
- Commence par demander à l’élève ce qu’il en pense.
- Si la réponse est partielle :
  - valide ce qui est juste,
  - signale calmement ce qui manque,
  - pose une question plus guidée.
- Si l’élève dit « je ne sais pas », « je ne comprends pas », ou répond très peu :
  - donne un INDICE, pas toute la réponse,
  - puis repose une question simple et ciblée.
- Ne donne une explication complète que si l’élève la demande clairement
  ou après plusieurs essais infructueux.

🔚 Fin de consultation :
- Quand toutes les étapes ont été vues, demande :
  « Pour être sûr, docteur : la téléconsultation est-elle terminée pour vous,
  ou souhaitez-vous ajouter quelque chose avant que je clôture ? »
- Si l’élève dit que c’est terminé, remercie-le et conclus :
  « Merci beaucoup docteur, je vais prendre soin de ma main. »
"""

REPORT_SYSTEM_PROMPT = """
Tu es un professeur de SVT.

À partir du dialogue ci-dessous entre un élève (médecin) et M. Dujardin (patient),
produis un bilan structuré en 4 parties, dans un style clair et concis.

Tu dois éviter les répétitions inutiles.
Tu n’écris la partie « 1. Identité des élèves » qu’UNE SEULE FOIS au début du rapport.
Très important : dans la partie 2, tu ne fais PAS un résumé.
Tu dois recopier les réponses de l’élève presque mot pour mot, en gardant sa formulation.

Structure attendue :

1. Identité des élèves
   - Prénoms
   - Classe

2. Réponses de l’élève
   - Reprends les phrases de l’élève telles qu’elles apparaissent dans le dialogue
     pour expliquer rougeur, chaleur, gonflement, douleur, cellules sentinelles,
     leucocytes et phagocytose.
   - Tu peux corriger UNIQUEMENT l’orthographe, les accords et la ponctuation.
   - Tu ne dois PAS changer le sens ni reformuler.
   - S’il existe plusieurs versions d’une même idée, choisis la version LA PLUS COMPLÈTE.

3. Corrections / Explications scientifiques

Dans cette partie, utilise exactement le texte ci-dessous, sans le modifier :

Rougeur : elle est due à la vasodilatation : les vaisseaux sanguins se dilatent et laissent passer plus de sang vers la zone blessée, ce qui la rend rouge.

Chaleur : le sang qui arrive en plus grande quantité est légèrement plus chaud que les tissus. Cet afflux de sang augmente la température de la zone, d’où la sensation de chaleur.

Gonflement : la vasodilatation rend les parois des capillaires plus perméables. Une partie du plasma sort des vaisseaux et s’accumule autour de la blessure : c’est l’œdème, qui fait gonfler les tissus.

Douleur : l’œdème distend la peau et appuie sur les récepteurs à la douleur ; de plus, des substances libérées lors de l’inflammation stimulent ces récepteurs. Ils envoient alors un message nerveux de douleur jusqu’au cerveau.

Cellules sentinelles : certaines cellules présentes dans la peau et les tissus (comme les mastocytes et les cellules dendritiques) reconnaissent l’entrée de microbes. Elles libèrent des médiateurs chimiques (histamine, chimiokines) qui provoquent la vasodilatation, l’augmentation de la perméabilité des capillaires et attirent d’autres cellules de défense vers la zone infectée.

Leucocytes : ce sont des globules blancs qui quittent les capillaires et se dirigent vers la zone infectée. Ils reconnaissent les microbes et participent à leur élimination.

Phagocytose : certains leucocytes, les phagocytes, capturent les microbes en plusieurs étapes :
- Reconnaissance et adhérence : le phagocyte reconnaît le microbe et s’y colle.
- Ingestion : il l’englobe dans une petite vésicule.
- Digestion : des enzymes digestives détruisent le microbe à l’intérieur de cette vésicule.
- Rejet des déchets : les débris du microbe qui n’ont pas été totalement digérés sont rejetés hors du phagocyte.

4. Bilan pédagogique
- Conclus par : « Très bien », « Bien » ou « À renforcer ».
- Rédige 1 à 2 phrases maximum sur la qualité de la consultation.

Rappels :
- N’écris les prénoms et la classe QU’UNE SEULE FOIS dans la section 1.
- N’invente aucune réponse de l’élève.
- Ne dépasse pas l’équivalent d’une page A4 en texte.
"""

INTRO_MESSAGE = (
    "Bonjour docteur, je me suis coupé hier en bricolant et ma main m’inquiète un peu. "
    "Elle est rouge, chaude, gonflée et douloureuse.\n\n"
    "Voici la photo de ma main.\n\n"
    "Pourquoi ma main est-elle rouge d’après vous ?"
)

# ================== OUTILS ==================

def call_openai(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.4,
    )
    return response.choices[0].message.content


def ensure_sessions_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def save_session(code, history, student_state, image_visible):
    """Sauvegarde locale. Sur Community Cloud, ce stockage n'est pas garanti à long terme."""
    ensure_sessions_dir()
    path = os.path.join(SESSIONS_DIR, f"{code}.json")
    data = {
        "history": history,
        "student_state": student_state,
        "image_visible": image_visible,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_session(code):
    path = os.path.join(SESSIONS_DIR, f"{code}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_conversation_text(history):
    conversation = ""
    tour = 0
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            tour += 1
            conversation += f"Tour {tour} - Élève : {content}\n"
        elif role == "assistant":
            conversation += f"Tour {tour} - M. Dujardin : {content}\n"
    return conversation or "Aucun dialogue."


def generate_report_text(history, student_state):
    conversation = build_conversation_text(history)
    name = (student_state or {}).get("name") or "élève"
    classe = (student_state or {}).get("class") or "classe"

    prompt_user = (
        f"Prénoms : {name}\n"
        f"Classe : {classe}\n\n"
        "Voici le dialogue complet entre l'élève (médecin) et M. Dujardin :\n\n"
        f"{conversation}"
    )

    return call_openai(
        [
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_user},
        ]
    )


# ================== PDF ==================

def pdf_safe_text(text):
    """Remplace quelques caractères Unicode non gérés par les polices PDF intégrées."""
    replacements = {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_report_pdf(report_text, student_state):
    """Construit le PDF entièrement en mémoire pour téléchargement direct."""
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="Bilan de téléconsultation SVT - M. Dujardin",
        author="Chatbot pédagogique SVT",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DujardinTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "DujardinBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=5,
    )
    heading_style = ParagraphStyle(
        "DujardinHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=8,
        spaceAfter=5,
    )

    story = [
        Paragraph("Bilan de téléconsultation SVT - M. Dujardin", title_style),
        Spacer(1, 4),
    ]

    safe_report = pdf_safe_text(report_text)

    for raw_line in safe_report.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 5))
            continue

        escaped = escape(line)

        # Titres 1. / 2. / 3. / 4.
        if line.startswith(("1.", "2.", "3.", "4.")):
            story.append(Paragraph(escaped, heading_style))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + escape(line[2:]), body_style))
        else:
            story.append(Paragraph(escaped, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ================== ÉTAT STREAMLIT ==================

def reset_state(message=None):
    st.session_state.history = [
        {"role": "assistant", "content": message or "Bonjour ! Quels sont vos prénoms ?"}
    ]
    st.session_state.student_state = {
        "step": "ask_name",
        "name": None,
        "class": None,
        "finished": False,
    }
    st.session_state.image_visible = False
    st.session_state.report_text = None
    st.session_state.report_pdf = None
    st.session_state.save_code = None


if "history" not in st.session_state:
    reset_state()

# ================== LOGIQUE CHAT ==================

def process_user_message(text):
    text = (text or "").strip()
    if not text:
        return

    upper = text.upper()
    lower = text.lower()
    history = st.session_state.history
    student_state = st.session_state.student_state

    # Commande RECOMMENCER
    if upper.startswith("RECOMMENCER"):
        reset_state("Consultation recommencée depuis le début. Bonjour ! Quels sont vos prénoms ?")
        st.rerun()

    # Commande REPRISE
    if upper.startswith("REPRISE"):
        parts = text.split()
        if len(parts) < 2:
            history.append({"role": "user", "content": text})
            history.append(
                {
                    "role": "assistant",
                    "content": "Pour reprendre une consultation, écris par exemple : REPRISE ABC123.",
                }
            )
            return

        code = parts[1].strip().upper()
        data = load_session(code)
        if data is None:
            history.append({"role": "user", "content": text})
            history.append(
                {
                    "role": "assistant",
                    "content": (
                        "Je ne trouve pas ce code de sauvegarde. "
                        "Vérifie le code ou recommence la consultation."
                    ),
                }
            )
            return

        st.session_state.history = data.get("history", [])
        st.session_state.student_state = data.get(
            "student_state",
            {"step": "ask_name", "name": None, "class": None, "finished": False},
        )
        st.session_state.image_visible = data.get("image_visible", False)
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": f"Consultation reprise avec le code {code}. Tu peux continuer là où tu t'étais arrêté.",
            }
        )
        return

    # Commande SAUVEGARDE
    if upper == "SAUVEGARDE":
        code = uuid.uuid4().hex[:6].upper()
        history.append({"role": "user", "content": text})
        save_session(
            code,
            history,
            student_state,
            st.session_state.image_visible,
        )
        st.session_state.save_code = code
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"La consultation est sauvegardée. Ton code est : {code}.\n\n"
                    f"Pour la reprendre : REPRISE {code}"
                ),
            }
        )
        return

    if "consultation est terminée" in lower:
        student_state["finished"] = True

    step = student_state.get("step", "ask_name")

    if step == "ask_name":
        student_state["name"] = text
        history.append({"role": "user", "content": text})
        history.append(
            {
                "role": "assistant",
                "content": "Merci ! Et dans quelle classe êtes-vous ? (ex : 3A, 3E...)",
            }
        )
        student_state["step"] = "ask_class"
        return

    if step == "ask_class":
        student_state["class"] = text
        history.append({"role": "user", "content": text})
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"Merci {student_state['name']} de la {student_state['class']}. "
                    "Nous pouvons commencer la consultation."
                ),
            }
        )
        history.append({"role": "assistant", "content": INTRO_MESSAGE})
        student_state["step"] = "consultation"
        st.session_state.image_visible = True
        return

    history.append({"role": "user", "content": text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        reply = call_openai(messages)
    except Exception as exc:
        reply = f"Erreur lors de l'appel à OpenAI : {exc}"

    history.append({"role": "assistant", "content": reply})

    if "je vais prendre soin de ma main" in reply.lower():
        student_state["finished"] = True


# ================== INTERFACE ==================

st.title("🤒 Chatbot M. Dujardin – Téléconsultation SVT (3e)")
st.caption("L'élève joue le rôle du médecin. M. Dujardin est le patient.")

with st.expander("ℹ️ Consignes et commandes", expanded=False):
    st.markdown(
        """
1. Indiquez vos prénoms, puis votre classe.
2. Répondez aux questions de M. Dujardin comme un médecin.
3. À la fin de la consultation, générez le bilan.
4. **SAUVEGARDE** : crée un code de reprise.
5. **REPRISE CODE** : reprend une consultation sauvegardée.
6. **RECOMMENCER** : recommence depuis le début.
        """
    )

left, right = st.columns([3, 1])

with left:
    for message in st.session_state.history:
        role = message.get("role")
        content = message.get("content", "")
        avatar = "🩺" if role == "user" else "🤒"
        label = "user" if role == "user" else "assistant"
        with st.chat_message(label, avatar=avatar):
            st.write(content)

with right:
    st.subheader("Photo")
    if st.session_state.image_visible and os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, caption="Photo de la main de M. Dujardin", use_container_width=True)
    elif os.path.exists(IMAGE_PATH):
        if st.button("Afficher la photo", use_container_width=True):
            st.session_state.image_visible = True
            st.rerun()
    else:
        st.info("Le fichier blessure_main.png devra être ajouté au dépôt GitHub.")

user_input = st.chat_input("Ta réponse / ta question")
if user_input:
    process_user_message(user_input)
    st.rerun()

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🧾 Générer le bilan final", use_container_width=True):
        finished = st.session_state.student_state.get("finished", False)

        nb_user_msgs = sum(
            1
            for m in st.session_state.history
            if isinstance(m, dict) and m.get("role") == "user"
        )

        if not finished:
            st.warning(
                "Le bilan ne peut être généré qu'à la fin de la téléconsultation. "
                "Laisse M. Dujardin terminer ou écris « la consultation est terminée »."
            )
        elif nb_user_msgs <= 2:
            st.warning(
                "Il faut au moins une réponse de médecin après les prénoms et la classe."
            )
        else:
            with st.spinner("Génération du bilan..."):
                try:
                    report_text = generate_report_text(
                        st.session_state.history,
                        st.session_state.student_state,
                    )
                    st.session_state.report_text = report_text
                    st.session_state.report_pdf = build_report_pdf(
                        report_text,
                        st.session_state.student_state,
                    )

                    code = uuid.uuid4().hex[:6].upper()
                    save_session(
                        code,
                        st.session_state.history,
                        st.session_state.student_state,
                        st.session_state.image_visible,
                    )
                    st.session_state.save_code = code
                except Exception as exc:
                    st.error(f"Impossible de générer le bilan : {exc}")

with col2:
    if st.button("🔄 Recommencer", use_container_width=True):
        reset_state()
        st.rerun()

if st.session_state.report_text:
    st.subheader("Bilan final")
    st.text_area(
        "Version texte copiable",
        st.session_state.report_text,
        height=420,
    )

    name = st.session_state.student_state.get("name") or "eleve"
    classe = st.session_state.student_state.get("class") or "classe"
    safe_filename = (
        f"bilan_dujardin_{name}_{classe}"
        .replace(" ", "_")
        .replace("/", "-")
    )

    if st.session_state.report_pdf:
        st.download_button(
            "📄 Télécharger le bilan en PDF",
            data=st.session_state.report_pdf,
            file_name=f"{safe_filename}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    if st.session_state.save_code:
        st.success(
            f"Code de sauvegarde de la consultation : {st.session_state.save_code}"
        )

st.caption(
    "Remarque : sur Streamlit Community Cloud, les fichiers de sauvegarde locaux "
    "peuvent être effacés lors d'un redémarrage de l'application. "
    "Le PDF, lui, est généré directement en mémoire puis téléchargé par l'utilisateur."
)
