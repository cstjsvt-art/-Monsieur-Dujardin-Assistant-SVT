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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# ================== CONFIG ==================

st.set_page_config(
    page_title="Chatbot M. Dujardin – Téléconsultation SVT",
    page_icon="🤒",
    layout="wide",
)

MODEL_NAME = "gpt-4o-mini"
IMAGE_PATH = "blessure_main.png"

# IMPORTANT :
# Pour une reprise fiable plusieurs jours plus tard, cette version utilise Upstash Redis.
# Ajouter dans les Secrets Streamlit :
# OPENAI_API_KEY = "..."
# UPSTASH_REDIS_REST_URL = "..."
# UPSTASH_REDIS_REST_TOKEN = "..."

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

try:
    UPSTASH_REDIS_REST_URL = st.secrets["UPSTASH_REDIS_REST_URL"]
except Exception:
    UPSTASH_REDIS_REST_URL = os.getenv("UPSTASH_REDIS_REST_URL")

try:
    UPSTASH_REDIS_REST_TOKEN = st.secrets["UPSTASH_REDIS_REST_TOKEN"]
except Exception:
    UPSTASH_REDIS_REST_TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN")

if not OPENAI_API_KEY:
    st.error(
        "La clé OPENAI_API_KEY est introuvable. "
        "Ajoute-la dans les Secrets de l'application Streamlit."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ================== STOCKAGE PERSISTANT ==================

redis_client = None

if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
    try:
        from upstash_redis import Redis
        redis_client = Redis(
            url=UPSTASH_REDIS_REST_URL,
            token=UPSTASH_REDIS_REST_TOKEN,
        )
    except Exception as exc:
        st.warning(
            "Le stockage persistant n'a pas pu être initialisé. "
            f"Détail : {exc}"
        )

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
2. Pour CHAQUE signe, lui faire expliquer la CAUSE :
   - rougeur : afflux sanguin / vasodilatation locale
   - chaleur : arrivée de sang plus chaud / augmentation du débit sanguin
   - gonflement : sortie de plasma, œdème, augmentation de la perméabilité des capillaires
   - douleur : stimulation des terminaisons nerveuses par l’œdème et les médiateurs chimiques
3. Amener l'élève à évoquer le rôle des cellules sentinelles :
   - elles détectent la présence de microbes,
   - elles libèrent des médiateurs chimiques (histamine, chimiokines) qui déclenchent la réaction inflammatoire.
4. Amener l'élève à parler du rôle des leucocytes / globules blancs recrutés grâce à ces signaux.
5. Amener l'élève à expliquer la phagocytose :
   - reconnaissance / adhésion
   - ingestion
   - digestion
   - rejet des déchets

📌 Persona & style :
- TA main est blessée, PAS celle du médecin.
- Tu dis toujours « ma main », « ma blessure », « ma coupure ».
- Tu ne dis JAMAIS « votre main » pour parler de la blessure.
- Tu parles comme un patient inquiet : « docteur », « ma main », etc.
- Tu te réfères naturellement à la photo de ta main.
- Tu restes bienveillant, simple et encourageant.

📌 Déroulement OBLIGATOIRE :
1. Rougeur
2. Chaleur
3. Gonflement
4. Douleur
5. Cellules sentinelles et médiateurs chimiques
6. Leucocytes / globules blancs
7. Phagocytose

Tu dois couvrir ces 7 points sans en sauter.

🔥 Gestion des réponses globales :
- Si l’élève donne plusieurs réponses d’un coup, tu ne lui demandes pas de tout réécrire.
- Tu t’appuies sur ce qu’il a déjà donné et tu vérifies les points manquants.
- Tu peux demander une précision ciblée si une idée est incomplète.

🔥 Très important :
- Même si l’élève parle spontanément des globules blancs ou de la phagocytose avant la douleur,
  tu NE SAUTES PAS l’étape douleur.
- Tu dois poser au moins une question explicite sur la douleur et sa cause.
- Tu dois poser au moins une question explicite sur les cellules sentinelles.
- Tu dois vérifier les 4 étapes de la phagocytose.

🧩 Aide progressive :
- D’abord demander à l’élève ce qu’il en pense.
- Si la réponse est partielle : valider ce qui est juste puis demander ce qui manque.
- Si l’élève ne sait pas : donner un indice, pas la réponse complète.
- Ne donner la réponse complète qu’après plusieurs essais infructueux ou si l’élève la demande clairement.

🔚 Fin de consultation :
- Tu ne termines la consultation que lorsque les 7 étapes ont été réellement abordées.
- Tu demandes :
  « Pour être sûr, docteur : la téléconsultation est-elle terminée pour vous,
  ou souhaitez-vous ajouter quelque chose avant que je clôture ? »
- Si l’élève confirme, tu conclus :
  « Merci beaucoup docteur, je vais prendre soin de ma main. »
"""

REPORT_SYSTEM_PROMPT = """
Tu es un professeur de SVT exigeant et juste.

À partir du dialogue entre un élève (médecin) et M. Dujardin (patient),
produis un bilan structuré.

RÈGLE ESSENTIELLE :
Tu dois évaluer uniquement ce que l'élève a réellement expliqué.
Ne valorise jamais une notion simplement parce que M. Dujardin l'a évoquée dans ses questions.

Structure attendue :

1. Identité
- Prénoms
- Classe
- Nombre de recommencements

2. Réponses de l'élève
Pour chaque notion ci-dessous, recopie la meilleure réponse réelle de l'élève, presque mot pour mot.
Tu peux corriger uniquement l'orthographe, les accords et la ponctuation.
Si une notion n'a pas été expliquée, écris : « Information absente ou insuffisante. »

Notions à vérifier :
- Rougeur
- Chaleur
- Gonflement
- Douleur
- Cellules sentinelles
- Médiateurs chimiques
- Leucocytes
- Phagocytose : reconnaissance / adhésion
- Phagocytose : ingestion
- Phagocytose : digestion
- Phagocytose : rejet des déchets

3. Corrections / explications scientifiques

Utilise exactement les explications suivantes :

Rougeur : elle est due à la vasodilatation : les vaisseaux sanguins se dilatent et laissent passer plus de sang vers la zone blessée, ce qui la rend rouge.

Chaleur : le sang qui arrive en plus grande quantité est légèrement plus chaud que les tissus. Cet afflux de sang augmente la température de la zone, d’où la sensation de chaleur.

Gonflement : la vasodilatation rend les parois des capillaires plus perméables. Une partie du plasma sort des vaisseaux et s’accumule autour de la blessure : c’est l’œdème, qui fait gonfler les tissus.

Douleur : l’œdème distend la peau et appuie sur les récepteurs à la douleur ; de plus, des substances libérées lors de l’inflammation stimulent ces récepteurs. Ils envoient alors un message nerveux de douleur jusqu’au cerveau.

Cellules sentinelles : certaines cellules présentes dans la peau et les tissus (comme les mastocytes et les cellules dendritiques) reconnaissent l’entrée de microbes. Elles libèrent des médiateurs chimiques (histamine, chimiokines) qui provoquent la vasodilatation, l’augmentation de la perméabilité des capillaires et attirent d’autres cellules de défense vers la zone infectée.

Leucocytes : ce sont des globules blancs qui quittent les capillaires et se dirigent vers la zone infectée. Ils reconnaissent les microbes et participent à leur élimination.

Phagocytose :
- Reconnaissance et adhésion : le phagocyte reconnaît le microbe et s’y colle.
- Ingestion : il l’englobe dans une petite vésicule.
- Digestion : des enzymes digestives détruisent le microbe à l’intérieur de cette vésicule.
- Rejet des déchets : les débris du microbe qui n’ont pas été totalement digérés sont rejetés hors du phagocyte.

4. Bilan pédagogique

Attribue impérativement UN SEUL niveau selon les règles ci-dessous :

TRÈS BIEN :
- toutes les notions principales sont présentes ;
- les 4 signes de l'inflammation sont expliqués avec leur mécanisme ;
- cellules sentinelles + médiateurs chimiques sont compris ;
- leucocytes sont expliqués ;
- les 4 étapes de la phagocytose sont présentes.
Quelques imprécisions mineures de vocabulaire sont tolérées.

BIEN :
- la majorité des notions est correcte ;
- au maximum 2 éléments importants sont absents ou incomplets ;
- aucune grande partie du raisonnement ne manque complètement.

À RENFORCER :
- 3 éléments importants ou plus sont absents ou insuffisants ;
OU
- un mécanisme majeur n'est pas compris ;
OU
- les cellules sentinelles / leucocytes / phagocytose sont très incomplètes ;
OU
- l'élève donne surtout des réponses très vagues.

Tu n'as PAS le droit d'écrire « Bien » si 3 éléments importants ou plus manquent.
Tu n'as PAS le droit d'écrire « Très bien » si une étape de la phagocytose manque.
Tu n'as PAS le droit d'écrire « Très bien » si cellules sentinelles ou médiateurs chimiques sont absents.

Termine par 1 ou 2 phrases maximum expliquant précisément ce qui est réussi et ce qui reste à améliorer.
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
        temperature=0.2,
    )
    return response.choices[0].message.content


def save_session(code, history, student_state, image_visible):
    if redis_client is None:
        return False, (
            "Le stockage persistant n'est pas configuré. "
            "Ajoute UPSTASH_REDIS_REST_URL et UPSTASH_REDIS_REST_TOKEN dans les Secrets Streamlit."
        )

    data = {
        "history": history,
        "student_state": student_state,
        "image_visible": image_visible,
    }

    redis_client.set(
        f"dujardin:{code}",
        json.dumps(data, ensure_ascii=False),
    )
    return True, None


def load_session(code):
    if redis_client is None:
        return None

    raw = redis_client.get(f"dujardin:{code}")
    if raw is None:
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    return json.loads(raw)


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
            # On n'intègre pas les commandes techniques dans le rapport scientifique.
            if content.upper() in {"SAUVEGARDE", "RECOMMENCER"} or content.upper().startswith("REPRISE "):
                continue
            tour += 1
            conversation += f"Tour {tour} - Élève : {content}\n"
        elif role == "assistant":
            conversation += f"Tour {tour} - M. Dujardin : {content}\n"

    return conversation or "Aucun dialogue."


def generate_report_text(history, student_state):
    conversation = build_conversation_text(history)

    first_names = (student_state or {}).get("first_names") or "élève"
    classe = (student_state or {}).get("class") or "classe"
    restart_count = int((student_state or {}).get("restart_count", 0))

    prompt_user = (
        f"Prénoms : {first_names}\n"
        f"Classe : {classe}\n"
        f"Nombre de recommencements : {restart_count}\n\n"
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


def build_report_pdf(report_text):
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
        fontSize=9.5,
        leading=13,
        spaceAfter=4,
    )

    heading_style = ParagraphStyle(
        "DujardinHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        spaceBefore=7,
        spaceAfter=4,
    )

    story = [
        Paragraph("Bilan de téléconsultation SVT - M. Dujardin", title_style),
        Spacer(1, 4),
    ]

    safe_report = pdf_safe_text(report_text)

    for raw_line in safe_report.splitlines():
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 4))
            continue

        if line.startswith(("1.", "2.", "3.", "4.")):
            story.append(Paragraph(escape(line), heading_style))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + escape(line[2:]), body_style))
        else:
            story.append(Paragraph(escape(line), body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ================== ÉTAT STREAMLIT ==================

def make_initial_student_state(restart_count=0):
    return {
        "step": "ask_first_names",
        "first_names": None,
        "class": None,
        "finished": False,
        "restart_count": restart_count,
    }


def reset_state(message=None, keep_restart_count=False):
    previous_count = 0
    if keep_restart_count and "student_state" in st.session_state:
        previous_count = int(st.session_state.student_state.get("restart_count", 0))

    st.session_state.history = [
        {
            "role": "assistant",
            "content": message or "Bonjour ! Quels sont vos prénoms ?",
        }
    ]
    st.session_state.student_state = make_initial_student_state(previous_count)
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

    # ---------- RECOMMENCER ----------
    if upper == "RECOMMENCER":
        new_count = int(student_state.get("restart_count", 0)) + 1

        st.session_state.history = [
            {
                "role": "assistant",
                "content": (
                    "Consultation recommencée depuis le début. "
                    "Bonjour ! Quels sont vos prénoms ?"
                ),
            }
        ]
        st.session_state.student_state = make_initial_student_state(new_count)
        st.session_state.image_visible = False
        st.session_state.report_text = None
        st.session_state.report_pdf = None
        st.session_state.save_code = None
        return

    # ---------- REPRISE ----------
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
                        "Vérifie le code. Si le problème persiste, demande de l'aide à ton professeur."
                    ),
                }
            )
            return

        st.session_state.history = data.get(
            "history",
            [{"role": "assistant", "content": "Bonjour ! Quels sont vos prénoms ?"}],
        )

        loaded_state = data.get("student_state") or make_initial_student_state()

        # Compatibilité avec une éventuelle ancienne sauvegarde
        if "name" in loaded_state and "first_names" not in loaded_state:
            loaded_state["first_names"] = loaded_state.pop("name")

        loaded_state.setdefault("restart_count", 0)
        loaded_state.setdefault("finished", False)

        st.session_state.student_state = loaded_state
        st.session_state.image_visible = data.get("image_visible", False)
        st.session_state.report_text = None
        st.session_state.report_pdf = None
        st.session_state.save_code = code

        st.session_state.history.append(
            {
                "role": "assistant",
                "content": (
                    f"Consultation reprise avec le code {code}. "
                    "Tu peux continuer là où tu t'étais arrêté."
                ),
            }
        )
        return

    # ---------- SAUVEGARDE ----------
    if upper == "SAUVEGARDE":
        code = uuid.uuid4().hex[:6].upper()

        ok, error = save_session(
            code,
            history,
            student_state,
            st.session_state.image_visible,
        )

        history.append({"role": "user", "content": text})

        if not ok:
            history.append(
                {
                    "role": "assistant",
                    "content": (
                        "La sauvegarde n'a pas pu être créée pour le moment. "
                        f"{error}"
                    ),
                }
            )
            return

        st.session_state.save_code = code
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"La consultation est sauvegardée. Ton code est : {code}\n\n"
                    f"Pour la reprendre plus tard, écris : REPRISE {code}"
                ),
            }
        )
        return

    step = student_state.get("step", "ask_first_names")

    # ---------- PRÉNOMS UNIQUEMENT ----------
    if step == "ask_first_names":
        student_state["first_names"] = text
        history.append({"role": "user", "content": text})
        history.append(
            {
                "role": "assistant",
                "content": "Merci ! Et dans quelle classe êtes-vous ? (ex : 3A, 3E...)",
            }
        )
        student_state["step"] = "ask_class"
        return

    # ---------- CLASSE ----------
    if step == "ask_class":
        student_state["class"] = text
        history.append({"role": "user", "content": text})
        history.append(
            {
                "role": "assistant",
                "content": (
                    f"Merci {student_state['first_names']} de la {student_state['class']}. "
                    "Nous pouvons commencer la consultation."
                ),
            }
        )
        history.append({"role": "assistant", "content": INTRO_MESSAGE})
        student_state["step"] = "consultation"
        st.session_state.image_visible = True
        return

    # ---------- CONSULTATION ----------
    history.append({"role": "user", "content": text})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        reply = call_openai(messages)
    except Exception as exc:
        reply = f"Erreur lors de l'appel à OpenAI : {exc}"

    history.append({"role": "assistant", "content": reply})

    # La fin n'est validée que si M. Dujardin prononce sa phrase finale.
    if "je vais prendre soin de ma main" in reply.lower():
        student_state["finished"] = True

# ================== INTERFACE ==================

st.title("🤒 Chatbot M. Dujardin – Téléconsultation SVT (3e)")
st.caption("L'élève joue le rôle du médecin. M. Dujardin est le patient.")

with st.expander("ℹ️ Consignes et commandes", expanded=False):
    st.markdown(
        """
1. Indiquez uniquement vos **prénoms**, puis votre classe.
2. Répondez aux questions de M. Dujardin comme un médecin.
3. Le bilan final ne peut être généré qu'une fois la téléconsultation réellement terminée.
4. **SAUVEGARDE** : crée un code de reprise.
5. **REPRISE CODE** : reprend une consultation sauvegardée.
6. **RECOMMENCER** : recommence depuis le début.
        """
    )

# Le dialogue utilise toute la largeur.
# La photo apparaît en petit format directement au moment où M. Dujardin la présente.
for message in st.session_state.history:
    role = message.get("role")
    content = message.get("content", "")
    avatar = "🩺" if role == "user" else "🤒"
    label = "user" if role == "user" else "assistant"

    with st.chat_message(label, avatar=avatar):
        st.write(content)

        if (
            role == "assistant"
            and st.session_state.image_visible
            and "Voici la photo de ma main." in content
        ):
            if os.path.exists(IMAGE_PATH):
                st.image(
                    IMAGE_PATH,
                    caption="Photo de la main de M. Dujardin",
                    width=320,
                )
            else:
                st.info("Le fichier blessure_main.png doit être ajouté au dépôt GitHub.")

# Petit rappel de la photo, sans occuper une colonne permanente.
if st.session_state.image_visible and os.path.exists(IMAGE_PATH):
    with st.expander("🔎 Revoir la photo de la main de M. Dujardin", expanded=False):
        st.image(
            IMAGE_PATH,
            caption="Photo de la main de M. Dujardin",
            width=320,
        )

st.caption(
    f"Recommencements : {int(st.session_state.student_state.get('restart_count', 0))}"
)

user_input = st.chat_input("Ta réponse / ta question")

if user_input:
    process_user_message(user_input)
    st.rerun()

st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("🧾 Générer le bilan final", use_container_width=True):
        finished = st.session_state.student_state.get("finished", False)

        nb_consultation_msgs = sum(
            1
            for m in st.session_state.history
            if isinstance(m, dict)
            and m.get("role") == "user"
            and (m.get("content") or "").strip().upper()
            not in {"SAUVEGARDE", "RECOMMENCER"}
            and not (m.get("content") or "").strip().upper().startswith("REPRISE ")
        )

        # Prénoms + classe = 2 messages.
        if not finished:
            st.warning(
                "Le bilan ne peut être généré qu'à la fin de la téléconsultation. "
                "M. Dujardin doit avoir terminé la consultation."
            )
        elif nb_consultation_msgs <= 2:
            st.warning(
                "Il faut au moins une réponse médicale après les prénoms et la classe."
            )
        else:
            with st.spinner("Génération du bilan..."):
                try:
                    report_text = generate_report_text(
                        st.session_state.history,
                        st.session_state.student_state,
                    )

                    st.session_state.report_text = report_text
                    st.session_state.report_pdf = build_report_pdf(report_text)

                except Exception as exc:
                    st.error(f"Impossible de générer le bilan : {exc}")

with col2:
    if st.button("🔄 Recommencer", use_container_width=True):
        st.session_state.student_state["restart_count"] = (
            int(st.session_state.student_state.get("restart_count", 0)) + 1
        )

        current_count = st.session_state.student_state["restart_count"]

        st.session_state.history = [
            {
                "role": "assistant",
                "content": (
                    "Consultation recommencée depuis le début. "
                    "Bonjour ! Quels sont vos prénoms ?"
                ),
            }
        ]
        st.session_state.student_state = make_initial_student_state(current_count)
        st.session_state.image_visible = False
        st.session_state.report_text = None
        st.session_state.report_pdf = None
        st.session_state.save_code = None
        st.rerun()

if st.session_state.report_text:
    st.subheader("Bilan final")

    st.text_area(
        "Version texte copiable",
        st.session_state.report_text,
        height=430,
    )

    first_names = st.session_state.student_state.get("first_names") or "eleve"
    classe = st.session_state.student_state.get("class") or "classe"

    safe_filename = (
        f"bilan_dujardin_{first_names}_{classe}"
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

if redis_client is None:
    st.info(
        "La sauvegarde persistante n'est pas encore configurée. "
        "Le chatbot fonctionne, mais SAUVEGARDE / REPRISE nécessitent Upstash Redis."
    )
else:
    st.caption("Sauvegarde persistante active.")
