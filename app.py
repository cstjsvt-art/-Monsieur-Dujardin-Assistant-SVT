import os
import uuid
import json
from io import BytesIO
from xml.sax.saxutils import escape

import streamlit as st
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chatbot M. Dujardin – Téléconsultation SVT",
    page_icon="🤒",
    layout="wide",
)

APP_VERSION = "M. Dujardin V3.5 - différenciation renforcée"
MODEL_NAME = "gpt-4o-mini"
IMAGE_PATH = "blessure_main.png"

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


try:
    ACCESS_CODES = list(st.secrets.get("ACCESS_CODES", []))
except Exception:
    ACCESS_CODES = []

if not OPENAI_API_KEY:
    st.error(
        "La clé OPENAI_API_KEY est introuvable. "
        "Ajoute-la dans les Secrets de l'application Streamlit."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# CONTRÔLE D'ACCÈS À L'APPLICATION
# ============================================================

def normalize_access_code(value):
    return (value or "").strip().upper()


VALID_ACCESS_CODES = {
    normalize_access_code(code)
    for code in ACCESS_CODES
    if str(code).strip()
}


def access_granted():
    return bool(st.session_state.get("access_granted", False))


def show_access_gate():
    st.title("🤒 Chatbot M. Dujardin – Téléconsultation SVT (3e)")
    st.caption("Accès réservé aux élèves disposant du code transmis par leur professeur.")
    st.caption(f"Version : {APP_VERSION}")

    st.info(
        "🔐 Entrez le code d'accès donné par votre professeur. "
        "Ce code est différent du code de reprise d'une consultation."
    )

    with st.form("access_form", clear_on_submit=False):
        entered_code = st.text_input(
            "Code d'accès",
            type="password",
            placeholder="Entrez le code d'accès…",
        )

        submit_access = st.form_submit_button(
            "Accéder à la téléconsultation",
            use_container_width=True,
            type="primary",
        )

    if submit_access:
        candidate = normalize_access_code(entered_code)

        if not VALID_ACCESS_CODES:
            st.error(
                "Aucun code d'accès n'est configuré dans les Secrets Streamlit. "
                "Le professeur doit renseigner ACCESS_CODES."
            )
        elif candidate in VALID_ACCESS_CODES:
            st.session_state.access_granted = True
            st.rerun()
        else:
            st.error(
                "Code incorrect. Vérifiez le code donné par votre professeur."
            )

    st.stop()


if not access_granted():
    show_access_gate()

# ============================================================
# STOCKAGE PERSISTANT
# ============================================================

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

# ============================================================
# PROMPT DU PATIENT
# ============================================================

SYSTEM_PROMPT = """
Tu joues EXCLUSIVEMENT le rôle de M. Dujardin, un patient adulte inquiet en téléconsultation.
L'élève joue le rôle du médecin.

Tu n'es PAS professeur : tu n'expliques pas les mécanismes toi-même.
Tu poses des QUESTIONS simples de patient pour amener l'élève à expliquer.

OBJECTIFS PÉDAGOGIQUES – NIVEAU 3e

1. Amener l'élève à identifier et expliquer les 4 signes de l'inflammation locale :
- rougeur : vasodilatation / afflux sanguin local ;
- chaleur : arrivée d'une plus grande quantité de sang plus chaud que les tissus ;
- gonflement : augmentation de la perméabilité des capillaires, sortie de plasma, œdème ;
- douleur : pression/étirement lié à l'œdème et action des médiateurs chimiques sur les récepteurs de la douleur.

2. Amener l'élève à expliquer le rôle des cellules sentinelles :
- elles détectent la présence de microbes ;
- elles libèrent des médiateurs chimiques qui déclenchent et organisent la réaction inflammatoire.

3. Amener l'élève à expliquer le rôle des leucocytes / globules blancs recrutés vers la zone lésée.

4. Amener l'élève à expliquer la phagocytose :
- reconnaissance / adhésion ;
- ingestion ;
- digestion ;
- rejet des déchets.

PERSONA
- TA main est blessée, PAS celle du médecin.
- Tu dis toujours « ma main », « ma blessure », « ma coupure ».
- Tu ne dis JAMAIS « votre main » pour parler de la blessure.
- Tu parles comme un patient inquiet et poli.
- Tu te réfères naturellement à la photo de ta main.
- Tu restes bienveillant et simple.

ORDRE OBLIGATOIRE
1. Rougeur
2. Chaleur
3. Gonflement
4. Douleur
5. Cellules sentinelles et médiateurs chimiques
6. Leucocytes / globules blancs
7. Phagocytose

Tu dois couvrir ces 7 points sans en sauter.

GESTION DES RÉPONSES GLOBALES
- Si l'élève donne plusieurs réponses d'un coup, ne lui demande pas de tout réécrire.
- Appuie-toi sur ce qu'il a déjà donné.
- Vérifie seulement les points encore manquants ou incomplets.

DIFFÉRENCIATION / AIDE PROGRESSIVE
- D'abord, demande ce que l'élève pense.
- Si la réponse est correcte et suffisamment complète pour le niveau 3e :
  1. valide ce qui est juste ;
  2. poursuis vers l'étape suivante sans exiger du vocabulaire plus expert.
- Si la réponse est correcte mais incomplète :
  1. valide UNIQUEMENT ce que l'élève a réellement dit ;
  2. NE DONNE PAS immédiatement le terme scientifique ou le mécanisme manquant ;
  3. pose d'abord UNE question de relance ciblée sur l'élément manquant.
- Si l'élève ne trouve pas après cette relance :
  1. donne un premier indice court, sans fournir la réponse ;
  2. laisse l'élève essayer à nouveau.
- Si l'élève ne trouve toujours pas :
  1. donne un second indice plus guidé ;
  2. laisse encore l'élève essayer.
- Ne donne l'explication complète qu'après plusieurs essais infructueux
  ou si l'élève la demande explicitement.
- Si l'élève dit « je ne sais pas », commence par un indice : ne donne jamais immédiatement la réponse complète.
- N'exige pas un vocabulaire parfait : l'objectif est la compréhension au niveau 3e.
- Exemple : si l'élève dit que du liquide sort des petits vaisseaux et s'accumule dans les tissus,
  demande d'abord ce qui pourrait permettre au liquide de traverser la paroi des capillaires.
  Ne donne pas immédiatement « augmentation de la perméabilité » ou « œdème ».
- Exemple : si l'élève explique que l'œdème appuie sur des récepteurs de la douleur,
  ne donne pas immédiatement le rôle des médiateurs chimiques ; demande d'abord s'il existe
  un autre facteur pouvant stimuler ces récepteurs.

TRÈS IMPORTANT
- Même si l'élève parle des globules blancs ou de la phagocytose avant la douleur,
  tu NE SAUTES PAS l'étape douleur.
- Tu poses au moins une question explicite sur la douleur et sa cause.
- Tu poses au moins une question explicite sur les cellules sentinelles.
- Tu vérifies les quatre étapes de la phagocytose.

FIN DE CONSULTATION
- Tu ne termines que lorsque les 7 étapes ont réellement été abordées.
- Tu demandes :
  « Pour être sûr, docteur : la téléconsultation est-elle terminée pour vous,
  ou souhaitez-vous ajouter quelque chose avant que je clôture ? »
- Si l'élève confirme que c'est terminé, tu conclus EXACTEMENT par :
  « Merci beaucoup docteur, je vais prendre soin de ma main. »
"""

INTRO_MESSAGE = (
    "Bonjour docteur, je me suis coupé hier en bricolant et ma main m'inquiète un peu. "
    "Elle est rouge, chaude, gonflée et douloureuse.\n\n"
    "Voici la photo de ma main.\n\n"
    "Pourquoi ma main est-elle rouge d'après vous ?"
)

# ============================================================
# RÉFÉRENCE SCIENTIFIQUE
# ============================================================

SCIENTIFIC_CORRECTIONS = [
    (
        "Rougeur",
        "Elle est due à la vasodilatation : les vaisseaux sanguins se dilatent et "
        "laissent passer plus de sang vers la zone blessée, ce qui la rend rouge."
    ),
    (
        "Chaleur",
        "Le sang qui arrive en plus grande quantité est légèrement plus chaud que les tissus. "
        "Cet afflux de sang augmente la température de la zone, d'où la sensation de chaleur."
    ),
    (
        "Gonflement",
        "La vasodilatation rend les parois des capillaires plus perméables. "
        "Une partie du plasma sort des vaisseaux et s'accumule autour de la blessure : "
        "c'est l'œdème, qui fait gonfler les tissus."
    ),
    (
        "Douleur",
        "L'œdème distend la peau et appuie sur les récepteurs à la douleur ; "
        "de plus, des substances libérées lors de l'inflammation stimulent ces récepteurs. "
        "Ils envoient alors un message nerveux de douleur jusqu'au cerveau."
    ),
    (
        "Cellules sentinelles",
        "Certaines cellules présentes dans la peau et les tissus (comme les mastocytes et "
        "les cellules dendritiques) reconnaissent l'entrée de microbes. Elles libèrent des "
        "médiateurs chimiques (histamine, chimiokines) qui provoquent la vasodilatation, "
        "l'augmentation de la perméabilité des capillaires et attirent d'autres cellules "
        "de défense vers la zone infectée."
    ),
    (
        "Leucocytes",
        "Ce sont des globules blancs qui quittent les capillaires et se dirigent vers la zone "
        "infectée. Ils reconnaissent les microbes et participent à leur élimination."
    ),
    (
        "Phagocytose",
        "Reconnaissance et adhésion : le phagocyte reconnaît le microbe et s'y colle. "
        "Ingestion : il l'englobe dans une petite vésicule. "
        "Digestion : des enzymes digestives détruisent le microbe à l'intérieur de cette vésicule. "
        "Rejet des déchets : les débris du microbe qui n'ont pas été totalement digérés sont "
        "rejetés hors du phagocyte."
    ),
]

# ============================================================
# ÉVALUATION PAR COMPÉTENCES
# ============================================================

COMPETENCY_NAMES = [
    "Expliquer les manifestations de la réaction inflammatoire",
    "Expliquer le rôle des cellules sentinelles et des médiateurs chimiques",
    "Expliquer le rôle des leucocytes dans la défense de l'organisme",
    "Expliquer les étapes de la phagocytose",
    "Communiquer à l'écrit en français dans un registre adapté au rôle de médecin",
]

LEVEL_LABELS = {
    1: "Niveau 1 – Maîtrise insuffisante",
    2: "Niveau 2 – Maîtrise fragile",
    3: "Niveau 3 – Maîtrise satisfaisante",
    4: "Niveau 4 – Très bonne maîtrise",
}

ASSESSMENT_SYSTEM_PROMPT = """
Tu es un professeur de SVT de collège qui évalue une téléconsultation réalisée par un élève de 3e.

Tu dois produire une évaluation PAR COMPÉTENCES, avec 4 niveaux.

ÉCHELLE
1 = Maîtrise insuffisante
Les bases ne sont pas assimilées ; les objectifs ne sont pas atteints.

2 = Maîtrise fragile
Les bases sont en cours d'acquisition ou partiellement comprises ;
l'élève a encore besoin d'un étayage important.

3 = Maîtrise satisfaisante
La compétence est acquise de manière autonome ;
c'est le niveau attendu en fin de cycle.

4 = Très bonne maîtrise
L'élève maîtrise très bien la compétence, mobilise un vocabulaire précis,
explique avec aisance et dépasse les attentes ordinaires de 3e.

Évalue EXACTEMENT ces 5 compétences :
1. Expliquer les manifestations de la réaction inflammatoire
2. Expliquer le rôle des cellules sentinelles et des médiateurs chimiques
3. Expliquer le rôle des leucocytes dans la défense de l'organisme
4. Expliquer les étapes de la phagocytose
5. Communiquer à l'écrit en français dans un registre adapté au rôle de médecin

RÈGLES D'ÉVALUATION SCIENTIFIQUE
- Évalue STRICTEMENT ce que l'élève a réellement écrit dans ses propres messages.
- Les explications, reformulations, mots scientifiques, corrections ou compléments donnés par M. Dujardin
  ne doivent JAMAIS être attribués à l'élève et ne doivent JAMAIS augmenter son niveau.
- Le dialogue complet sert uniquement à mesurer le degré d'aide et d'étayage reçu.
- Une notion partiellement correcte ne doit pas être traitée comme absente.
- Ne sois ni trop généreux, ni excessivement sévère.
- Quelques fautes de vocabulaire ou formulations maladroites sont compatibles avec le niveau 3
  si le mécanisme est globalement compris.
- Le niveau 3 correspond au niveau attendu en 3e : l'élève comprend le mécanisme essentiel
  et sait l'expliquer de façon autonome, même s'il ne cite pas tous les termes spécialisés.
- Ne baisse PAS au niveau 2 uniquement parce que l'élève ne cite pas « histamine », « chimiokines »,
  « diapédèse », « phagosome » ou un autre terme spécialisé si le mécanisme attendu est correctement compris.
- Pour les cellules sentinelles / médiateurs chimiques, le niveau 3 est justifié si l'élève explique de lui-même
  que les cellules sentinelles détectent les microbes, libèrent des substances chimiques et que ces substances
  déclenchent/organisent la réaction inflammatoire ou attirent des cellules de défense.
- Le niveau 4 suppose une réponse particulièrement précise, complète et autonome,
  avec un vocabulaire scientifique riche et une mobilisation allant au-delà des attentes ordinaires de 3e.
- Le niveau 1 doit être réservé aux bases réellement non comprises ou absentes.
- Le niveau 2 correspond à une compréhension partielle, fragile ou obtenue grâce à un étayage important.
- Dans chaque justification, cite seulement des éléments réellement formulés par l'élève.

LANGUE FRANÇAISE / REGISTRE DU MÉDECIN
- N'évalue pas seulement l'orthographe.
- Observe surtout si l'élève écrit de façon compréhensible, avec des phrases adaptées,
  un vocabulaire approprié et un registre compatible avec le rôle d'un médecin.
- Quelques fautes d'orthographe n'empêchent pas le niveau 3.
- Un ton très familier, des formulations de type discussion entre amis,
  des insultes, du langage SMS ou des réponses extrêmement relâchées doivent faire baisser le niveau.
- Le niveau 4 suppose une expression particulièrement claire, structurée et professionnelle pour un élève de 3e.

PRISE EN COMPTE DE L'AIDE
Le dialogue contient les relances et indices de M. Dujardin.
- Une réponse obtenue seulement après beaucoup d'étayage peut correspondre au niveau 2.
- Une réponse trouvée avec une petite relance peut rester au niveau 3.
- Le niveau 4 suppose une forte autonomie.

RENVOIE UNIQUEMENT un objet JSON valide, sans markdown, sans texte avant ou après :

{
  "competences": [
    {
      "nom": "Expliquer les manifestations de la réaction inflammatoire",
      "niveau": 3,
      "justification": "..."
    },
    {
      "nom": "Expliquer le rôle des cellules sentinelles et des médiateurs chimiques",
      "niveau": 2,
      "justification": "..."
    },
    {
      "nom": "Expliquer le rôle des leucocytes dans la défense de l'organisme",
      "niveau": 3,
      "justification": "..."
    },
    {
      "nom": "Expliquer les étapes de la phagocytose",
      "niveau": 4,
      "justification": "..."
    },
    {
      "nom": "Communiquer à l'écrit en français dans un registre adapté au rôle de médecin",
      "niveau": 3,
      "justification": "..."
    }
  ],
  "appreciation": "2 ou 3 phrases maximum."
}
"""

# ============================================================
# OUTILS OPENAI
# ============================================================

def call_openai(messages, temperature=0.2):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

# ============================================================
# SAUVEGARDE / REPRISE
# ============================================================

def save_session(code, history, student_state, image_visible):
    if redis_client is None:
        return False, (
            "Le stockage persistant n'est pas configuré. "
            "Ajoute UPSTASH_REDIS_REST_URL et UPSTASH_REDIS_REST_TOKEN "
            "dans les Secrets Streamlit."
        )

    data = {
        "history": history,
        "student_state": student_state,
        "image_visible": image_visible,
    }

    try:
        redis_client.set(
            f"dujardin:{code}",
            json.dumps(data, ensure_ascii=False),
        )
        return True, None
    except Exception as exc:
        return False, (
            "La sauvegarde persistante est momentanément indisponible. "
            "Vérifie les paramètres Upstash dans les Secrets Streamlit. "
            f"Détail technique : {exc.__class__.__name__}"
        )


def autosave_current_session():
    code = st.session_state.get("save_code")
    if not code:
        return

    save_session(
        code,
        st.session_state.history,
        st.session_state.student_state,
        st.session_state.image_visible,
    )


def load_session(code):
    if redis_client is None:
        return None

    try:
        raw = redis_client.get(f"dujardin:{code}")
    except Exception:
        return None

    if raw is None:
        return None

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    try:
        return json.loads(raw)
    except Exception:
        return None

# ============================================================
# DIALOGUE DU RAPPORT
# ============================================================

def is_technical_command(content):
    upper = (content or "").strip().upper()
    return (
        upper == "SAUVEGARDE"
        or upper == "RECOMMENCER"
        or upper.startswith("REPRISE ")
    )


def get_consultation_dialogue(history):
    dialogue = []
    consultation_started = False

    for msg in history:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role")
        content = (msg.get("content") or "").strip()

        if not content or is_technical_command(content):
            continue

        if role == "assistant" and content == INTRO_MESSAGE:
            consultation_started = True

        if consultation_started:
            dialogue.append({
                "role": role,
                "content": content,
            })

    return dialogue


def build_dialogue_text(history):
    dialogue = get_consultation_dialogue(history)

    if not dialogue:
        return "Aucun dialogue de téléconsultation."

    lines = []

    for msg in dialogue:
        speaker = "Élève (médecin)" if msg["role"] == "user" else "M. Dujardin"
        lines.append(f"{speaker} : {msg['content']}")

    return "\n".join(lines)


def build_student_only_text(history):
    """Source principale pour noter : uniquement les productions de l'élève."""
    dialogue = get_consultation_dialogue(history)

    student_messages = [
        msg["content"]
        for msg in dialogue
        if msg.get("role") == "user"
    ]

    if not student_messages:
        return "Aucune production de l'élève."

    return "\n".join(
        f"Réponse élève {index} : {content}"
        for index, content in enumerate(student_messages, start=1)
    )

# ============================================================
# ÉVALUATION
# ============================================================

def normalize_assessment(data):
    result = {
        "competences": [],
        "appreciation": str(data.get("appreciation", "")).strip(),
    }

    returned = data.get("competences", [])
    by_name = {}

    for item in returned:
        if not isinstance(item, dict):
            continue

        name = str(item.get("nom", "")).strip()

        try:
            level = int(item.get("niveau", 1))
        except Exception:
            level = 1

        level = min(4, max(1, level))
        justification = str(item.get("justification", "")).strip()

        by_name[name] = {
            "nom": name,
            "niveau": level,
            "justification": justification,
        }

    for expected_name in COMPETENCY_NAMES:
        item = by_name.get(expected_name)

        if item is None:
            item = {
                "nom": expected_name,
                "niveau": 1,
                "justification": "Évaluation automatique indisponible pour cette compétence.",
            }

        result["competences"].append(item)

    if not result["appreciation"]:
        result["appreciation"] = (
            "Le bilan met en évidence les acquis et les points à consolider au cours de la téléconsultation."
        )

    return result


def generate_assessment(history):
    dialogue_text = build_dialogue_text(history)
    student_only_text = build_student_only_text(history)

    raw = call_openai(
        [
            {"role": "system", "content": ASSESSMENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "SOURCE PRINCIPALE POUR NOTER : PRODUCTIONS DE L'ÉLÈVE UNIQUEMENT.\n"
                    "Attribue les acquis uniquement à partir de cette partie :\n\n"
                    f"{student_only_text}\n\n"
                    "CONTEXTE SECONDAIRE : DIALOGUE COMPLET.\n"
                    "Utilise cette partie seulement pour mesurer les relances et l'aide de M. Dujardin.\n"
                    "N'attribue jamais à l'élève une information formulée uniquement par M. Dujardin.\n\n"
                    f"{dialogue_text}"
                ),
            },
        ],
        temperature=0.0,
    )

    cleaned = raw.strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
    except Exception:
        raise RuntimeError(
            "L'évaluation automatique n'a pas renvoyé un JSON valide. "
            "Relance la génération du bilan."
        )

    return normalize_assessment(data)

# ============================================================
# RAPPORT TEXTE
# ============================================================

def build_report_text(student_state, history, assessment):
    first_names = (student_state or {}).get("first_names") or "élève"
    classe = (student_state or {}).get("class") or "classe"
    restart_count = int((student_state or {}).get("restart_count", 0))

    lines = []

    lines.append("1. Identité")
    lines.append(f"- Prénoms : {first_names}")
    lines.append(f"- Classe : {classe}")
    lines.append(f"- Nombre de recommencements : {restart_count}")
    lines.append("")

    lines.append("2. Dialogue complet de la téléconsultation")
    dialogue = get_consultation_dialogue(history)

    for msg in dialogue:
        speaker = "Élève (médecin)" if msg["role"] == "user" else "M. Dujardin"
        lines.append(f"{speaker} : {msg['content']}")
        lines.append("")

    lines.append("3. Corrections / explications scientifiques")

    for title, explanation in SCIENTIFIC_CORRECTIONS:
        lines.append(f"{title} : {explanation}")
        lines.append("")

    lines.append("4. Bilan pédagogique par compétences")

    for comp in assessment["competences"]:
        level = comp["niveau"]
        lines.append(f"- {comp['nom']} : {LEVEL_LABELS[level]}")
        lines.append(f"  Justification : {comp['justification']}")

    lines.append("")
    lines.append("5. Appréciation")
    lines.append(assessment["appreciation"])

    return "\n".join(lines)

# ============================================================
# PDF
# ============================================================

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


def build_report_pdf(student_state, history, assessment):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
        title="Bilan de téléconsultation SVT - M. Dujardin",
        author="Chatbot pédagogique SVT",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DujardinTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    heading_style = ParagraphStyle(
        "DujardinHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        spaceBefore=8,
        spaceAfter=5,
    )

    body_style = ParagraphStyle(
        "DujardinBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        spaceAfter=4,
    )

    small_style = ParagraphStyle(
        "DujardinSmall",
        parent=body_style,
        fontSize=8,
        leading=10,
    )

    story = [
        Paragraph("Bilan de téléconsultation SVT - M. Dujardin", title_style),
    ]

    first_names = (student_state or {}).get("first_names") or "élève"
    classe = (student_state or {}).get("class") or "classe"
    restart_count = int((student_state or {}).get("restart_count", 0))

    story.append(Paragraph("1. Identité", heading_style))
    story.append(
        Paragraph(
            f"<b>Prénoms :</b> {escape(pdf_safe_text(first_names))}",
            body_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Classe :</b> {escape(pdf_safe_text(classe))}",
            body_style,
        )
    )
    story.append(
        Paragraph(
            f"<b>Nombre de recommencements :</b> {restart_count}",
            body_style,
        )
    )

    story.append(
        Paragraph("2. Dialogue complet de la téléconsultation", heading_style)
    )

    for msg in get_consultation_dialogue(history):
        speaker = "Élève (médecin)" if msg["role"] == "user" else "M. Dujardin"
        content = escape(pdf_safe_text(msg["content"])).replace("\n", "<br/>")

        story.append(
            Paragraph(
                f"<b>{escape(speaker)} :</b> {content}",
                body_style,
            )
        )

    story.append(
        Paragraph("3. Corrections / explications scientifiques", heading_style)
    )

    for title, explanation in SCIENTIFIC_CORRECTIONS:
        story.append(
            Paragraph(
                f"<b>{escape(title)} :</b> {escape(pdf_safe_text(explanation))}",
                body_style,
            )
        )

    story.append(
        Paragraph("4. Bilan pédagogique par compétences", heading_style)
    )

    table_data = [
        [
            Paragraph("<b>Compétence évaluée</b>", small_style),
            Paragraph("<b>Niveau</b>", small_style),
            Paragraph("<b>Justification</b>", small_style),
        ]
    ]

    level_backgrounds = {
        1: colors.HexColor("#FDE2E2"),
        2: colors.HexColor("#FFF0D6"),
        3: colors.HexColor("#E2F5E9"),
        4: colors.HexColor("#E1ECFA"),
    }

    row_levels = []

    for comp in assessment["competences"]:
        level = int(comp["niveau"])
        row_levels.append(level)

        level_text = LEVEL_LABELS[level].split("–", 1)[1].strip()

        table_data.append(
            [
                Paragraph(
                    escape(pdf_safe_text(comp["nom"])),
                    small_style,
                ),
                Paragraph(
                    f"<b>Niveau {level}</b><br/>{escape(level_text)}",
                    small_style,
                ),
                Paragraph(
                    escape(pdf_safe_text(comp["justification"])),
                    small_style,
                ),
            ]
        )

    table = Table(
        table_data,
        colWidths=[6.2 * cm, 3.2 * cm, 8.0 * cm],
        repeatRows=1,
        hAlign="LEFT",
    )

    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDEDED")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]

    for row_index, level in enumerate(row_levels, start=1):
        commands.append(
            (
                "BACKGROUND",
                (1, row_index),
                (1, row_index),
                level_backgrounds[level],
            )
        )

    table.setStyle(TableStyle(commands))
    story.append(table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Repères des niveaux :</b>", body_style))
    story.append(
        Paragraph(
            "<b>Niveau 1 – Maîtrise insuffisante :</b> bases non assimilées, objectifs non atteints.",
            small_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Niveau 2 – Maîtrise fragile :</b> bases partiellement comprises, besoin d'un étayage important.",
            small_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Niveau 3 – Maîtrise satisfaisante :</b> compétence acquise de manière autonome, niveau attendu.",
            small_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Niveau 4 – Très bonne maîtrise :</b> maîtrise très solide, précise et mobilisée avec aisance.",
            small_style,
        )
    )

    story.append(Paragraph("5. Appréciation", heading_style))
    story.append(
        Paragraph(
            escape(pdf_safe_text(assessment["appreciation"])),
            body_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ============================================================
# ÉTAT STREAMLIT
# ============================================================

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
        previous_count = int(
            st.session_state.student_state.get("restart_count", 0)
        )

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
    st.session_state.assessment = None
    st.session_state.save_code = None


if "history" not in st.session_state:
    reset_state()


def restart_consultation():
    """Recommence depuis le début en conservant le code de sauvegarde actif."""
    current_count = int(
        st.session_state.student_state.get("restart_count", 0)
    ) + 1
    active_code = st.session_state.get("save_code")

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
    st.session_state.assessment = None
    st.session_state.save_code = active_code
    st.session_state.restart_confirm = False

    autosave_current_session()


# ============================================================
# LOGIQUE DU CHAT
# ============================================================

def process_user_message(text):
    text = (text or "").strip()

    if not text:
        return

    upper = text.upper()
    history = st.session_state.history
    student_state = st.session_state.student_state

    # ---------- RECOMMENCER ----------
    # Compatibilité avec l'ancienne commande texte.
    if upper == "RECOMMENCER":
        restart_consultation()
        return

    # ---------- REPRISE ----------
    if upper.startswith("REPRISE"):
        parts = text.split()

        if len(parts) < 2:
            history.append({"role": "user", "content": text})
            history.append(
                {
                    "role": "assistant",
                    "content": (
                        "Pour reprendre une consultation, écris par exemple : "
                        "REPRISE ABC123."
                    ),
                }
            )
            return

        # Accepte les deux formes :
        # REPRISE ABC123
        # REPRISE CODE ABC123
        if len(parts) >= 3 and parts[1].strip().upper() == "CODE":
            code = parts[2].strip().upper()
        else:
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
            [
                {
                    "role": "assistant",
                    "content": "Bonjour ! Quels sont vos prénoms ?",
                }
            ],
        )

        loaded_state = data.get("student_state") or make_initial_student_state()

        if "name" in loaded_state and "first_names" not in loaded_state:
            loaded_state["first_names"] = loaded_state.pop("name")

        loaded_state.setdefault("restart_count", 0)
        loaded_state.setdefault("finished", False)

        st.session_state.student_state = loaded_state
        st.session_state.image_visible = data.get("image_visible", False)
        st.session_state.report_text = None
        st.session_state.report_pdf = None
        st.session_state.assessment = None
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
        code = st.session_state.get("save_code")

        if not code:
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
                    f"Pour la reprendre à la prochaine séance, écris : REPRISE {code}\n\n"
                    "Cette sauvegarde sera mise à jour automatiquement pendant la suite de la consultation."
                ),
            }
        )
        return

    step = student_state.get("step", "ask_first_names")

    # ---------- PRÉNOMS ----------
    if step == "ask_first_names":
        student_state["first_names"] = text

        history.append(
            {
                "role": "user",
                "content": text,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": "Merci ! Et dans quelle classe êtes-vous ? (ex : 3A, 3E...)",
            }
        )

        student_state["step"] = "ask_class"
        autosave_current_session()
        return

    # ---------- CLASSE ----------
    if step == "ask_class":
        student_state["class"] = text

        history.append(
            {
                "role": "user",
                "content": text,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": (
                    f"Merci {student_state['first_names']} de la {student_state['class']}. "
                    "Nous pouvons commencer la consultation."
                ),
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": INTRO_MESSAGE,
            }
        )

        student_state["step"] = "consultation"
        st.session_state.image_visible = True

        autosave_current_session()
        return

    # ---------- CONSULTATION ----------
    history.append(
        {
            "role": "user",
            "content": text,
        }
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ] + history

    try:
        reply = call_openai(
            messages,
            temperature=0.35,
        )
    except Exception as exc:
        reply = f"Erreur lors de l'appel à OpenAI : {exc}"

    history.append(
        {
            "role": "assistant",
            "content": reply,
        }
    )

    if "merci beaucoup docteur, je vais prendre soin de ma main" in reply.lower():
        student_state["finished"] = True

    autosave_current_session()

# ============================================================
# INTERFACE
# ============================================================

st.title("🤒 Chatbot M. Dujardin – Téléconsultation SVT (3e)")
with st.sidebar:
    st.markdown("### Accès")

    if st.button("🔒 Quitter l'accès protégé", use_container_width=True):
        st.session_state.access_granted = False
        st.rerun()

    if not st.session_state.get("restart_confirm", False):
        if st.button("🔄 Recommencer la consultation", use_container_width=True):
            st.session_state.restart_confirm = True
            st.rerun()
    else:
        st.warning(
            "Cette action effacera la consultation en cours et recommencera depuis le début."
        )

        confirm_col, cancel_col = st.columns(2)

        with confirm_col:
            if st.button(
                "✅ Confirmer",
                use_container_width=True,
                type="primary",
                key="confirm_restart_sidebar",
            ):
                restart_consultation()
                st.rerun()

        with cancel_col:
            if st.button(
                "Annuler",
                use_container_width=True,
                key="cancel_restart_sidebar",
            ):
                st.session_state.restart_confirm = False
                st.rerun()

st.caption("L'élève joue le rôle du médecin. M. Dujardin est le patient.")
st.caption(f"Version : {APP_VERSION}")

with st.expander("ℹ️ Consignes et commandes", expanded=False):
    st.markdown(
        """
1. Indiquez uniquement vos **prénoms**, puis votre classe.
2. Répondez aux questions de M. Dujardin comme un médecin.
3. Le bouton **Recommencer la consultation**, dans la barre latérale, permet de repartir depuis le début après confirmation.
4. Le bilan final ne peut être généré qu'une fois la téléconsultation réellement terminée.
5. Pour interrompre une séance et la reprendre plus tard, écrivez **SAUVEGARDE** dans la zone de réponse puis conservez le code obtenu.
6. À la séance suivante, écrivez **REPRISE** suivi de votre code, par exemple **REPRISE FE0A2F**.
7. Le **code d'accès à l'application** et le **code de reprise d'une consultation** sont deux codes différents.
        """
    )

# ---------- DIALOGUE ----------
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
                st.info(
                    "Le fichier blessure_main.png doit être ajouté au dépôt GitHub."
                )

# ---------- ZONE DE RÉPONSE INLINE ----------
# IMPORTANT : on n'utilise PAS st.chat_input.
# Le formulaire reste exactement sous la dernière question.

if not st.session_state.student_state.get("finished", False):
    st.markdown(
        """
        <style>
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.form("response_form", clear_on_submit=True):
        st.markdown("#### 🩺 Votre réponse")

        user_input = st.text_input(
            "Réponse",
            placeholder="Écrivez votre réponse ici puis appuyez sur Entrée…",
            label_visibility="collapsed",
        )

        send_clicked = st.form_submit_button("Envoyer")

    if send_clicked and user_input.strip():
        process_user_message(user_input)
        st.rerun()

else:
    st.success(
        "✅ La téléconsultation est terminée. "
        "Vous pouvez maintenant générer votre bilan et le télécharger en PDF."
    )

    if st.button(
        "🧾 Générer le bilan final",
        use_container_width=True,
        type="primary",
    ):
        with st.spinner("Analyse des compétences et génération du bilan..."):
            try:
                assessment = generate_assessment(
                    st.session_state.history
                )

                report_text = build_report_text(
                    st.session_state.student_state,
                    st.session_state.history,
                    assessment,
                )

                report_pdf = build_report_pdf(
                    st.session_state.student_state,
                    st.session_state.history,
                    assessment,
                )

                st.session_state.assessment = assessment
                st.session_state.report_text = report_text
                st.session_state.report_pdf = report_pdf

                autosave_current_session()

            except Exception as exc:
                st.error(
                    f"Impossible de générer le bilan : {exc}"
                )

st.divider()

# ============================================================
# AFFICHAGE DU BILAN
# ============================================================

if st.session_state.report_text:
    st.subheader("Bilan final")

    if st.session_state.assessment:
        st.markdown("### Bilan pédagogique par compétences")

        for comp in st.session_state.assessment["competences"]:
            level = comp["niveau"]

            if level == 1:
                icon = "🔴"
            elif level == 2:
                icon = "🟠"
            elif level == 3:
                icon = "🟢"
            else:
                icon = "🔵"

            st.markdown(
                f"**{icon} {comp['nom']} — {LEVEL_LABELS[level]}**  \n"
                f"{comp['justification']}"
            )

        st.markdown(
            f"**Appréciation :** "
            f"{st.session_state.assessment['appreciation']}"
        )

    with st.expander(
        "Afficher le rapport complet en version texte",
        expanded=False,
    ):
        st.text_area(
            "Version texte copiable",
            st.session_state.report_text,
            height=520,
        )

    first_names = (
        st.session_state.student_state.get("first_names")
        or "eleve"
    )

    classe = (
        st.session_state.student_state.get("class")
        or "classe"
    )

    safe_filename = (
        f"bilan_dujardin_{first_names}_{classe}"
        .replace(" ", "_")
        .replace("/", "-")
    )

    if st.session_state.report_pdf:
        st.download_button(
            "📄 Télécharger le bilan complet en PDF",
            data=st.session_state.report_pdf,
            file_name=f"{safe_filename}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# ============================================================
# ÉTAT DE LA SAUVEGARDE
# ============================================================

if redis_client is None:
    st.info(
        "La sauvegarde persistante n'est pas encore configurée. "
        "Le chatbot fonctionne, mais SAUVEGARDE / REPRISE nécessitent Upstash Redis."
    )
else:
    if st.session_state.get("save_code"):
        st.caption(
            f"💾 Sauvegarde automatique active — code de reprise : "
            f"{st.session_state.save_code}"
        )
    else:
        st.caption(
            "💾 Sauvegarde persistante disponible. "
            "Écrivez SAUVEGARDE dans la zone de réponse pour obtenir un code ; "
            "la consultation sera ensuite mise à jour automatiquement."
        )
