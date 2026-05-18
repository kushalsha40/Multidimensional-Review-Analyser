import requests
import streamlit as st


API_URL = "http://localhost:8000/analyze"
DEFAULT_REVIEW = (
    "The story was strong and the acting was excellent, but the ending felt a "
    "little disappointing."
)


st.set_page_config(
    page_title="IMDB Review Analysis",
    page_icon="🎬",
    layout="wide",
)


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(211, 84, 0, 0.18), transparent 28%),
                radial-gradient(circle at top left, rgba(22, 160, 133, 0.14), transparent 24%),
                linear-gradient(180deg, #f8f2e8 0%, #f3ede4 100%);
        }

        .hero-card,
        .panel-card,
        .metric-card,
        .similarity-card {
            background: rgba(255, 252, 247, 0.86);
            border: 1px solid rgba(81, 58, 42, 0.10);
            border-radius: 22px;
            box-shadow: 0 14px 32px rgba(73, 48, 28, 0.08);
        }

        .hero-card {
            padding: 28px 30px 26px 30px;
            margin-bottom: 18px;
        }

        .panel-card {
            padding: 22px 24px;
            margin-top: 12px;
        }

        .metric-card {
            padding: 18px 18px 14px 18px;
            min-height: 112px;
        }

        .similarity-card {
            padding: 18px 18px 10px 18px;
            margin-bottom: 14px;
        }

        .eyebrow {
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.76rem;
            color: #996a43;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .hero-title {
            font-size: 2.5rem;
            line-height: 1.05;
            color: #2c1d14;
            font-weight: 800;
            margin: 0 0 10px 0;
        }

        .hero-copy {
            font-size: 1.02rem;
            color: #5a4332;
            line-height: 1.6;
            margin: 0;
        }

        .section-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #342218;
            margin-bottom: 14px;
        }

        .metric-label {
            color: #8a6448;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .metric-value {
            color: #2d1c12;
            font-size: 1.6rem;
            font-weight: 800;
            margin-bottom: 6px;
        }

        .metric-subtext {
            color: #6d5647;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        .badge {
            display: inline-block;
            padding: 8px 14px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
            margin-top: 2px;
        }

        .badge-positive {
            color: #0f5132;
            background: rgba(25, 135, 84, 0.16);
            border: 1px solid rgba(25, 135, 84, 0.24);
        }

        .badge-negative {
            color: #842029;
            background: rgba(220, 53, 69, 0.14);
            border: 1px solid rgba(220, 53, 69, 0.20);
        }

        .badge-neutral {
            color: #664d03;
            background: rgba(255, 193, 7, 0.18);
            border: 1px solid rgba(255, 193, 7, 0.25);
        }

        .review-label {
            color: #8a6448;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .review-body {
            color: #37271b;
            line-height: 1.7;
            font-size: 0.98rem;
        }

        .small-note {
            color: #7a6454;
            font-size: 0.88rem;
        }

        .score-chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(72, 52, 212, 0.08);
            color: #4a2f1b;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Movie Language Lab</div>
            <div class="hero-title">Read The Mood Behind A Review</div>
            <p class="hero-copy">
                Paste any movie review and get sentiment, emotion distribution,
                writing-style signals, aspect-level feedback, and the closest
                reviews from your embedding index in one place.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label, value, subtext):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sentiment(sentiment):
    sentiment_text = (sentiment or "unknown").capitalize()
    if sentiment == "positive":
        badge_class = "badge badge-positive"
    elif sentiment == "negative":
        badge_class = "badge badge-negative"
    else:
        badge_class = "badge badge-neutral"

    st.markdown(
        f"""
        <div class="panel-card">
            <div class="section-title">Sentiment</div>
            <span class="{badge_class}">{sentiment_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_emotions(emotion_scores):
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Emotion Profile</div>', unsafe_allow_html=True)

    if not emotion_scores:
        st.info("No emotion scores available.")
    else:
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        render_metric_card(
            "Dominant Emotion",
            dominant_emotion.capitalize(),
            f"Highest score: {emotion_scores.get(dominant_emotion, 0):.4f}",
        )
        for emotion, score in sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True):
            st.write(f"{emotion.capitalize()}  |  {score:.4f}")
            st.progress(min(max(float(score), 0.0), 1.0))

    st.markdown("</div>", unsafe_allow_html=True)


def render_style(style_scores):
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Writing Style</div>', unsafe_allow_html=True)

    if not style_scores:
        st.info("No writing-style metrics available.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card(
                "Sentence Length",
                style_scores.get("sentence_length", 0),
                "Approximate token count after preprocessing.",
            )
        with col2:
            render_metric_card(
                "Average Word Length",
                f'{style_scores.get("avg_word_length", 0):.2f}',
                "Longer words often indicate denser phrasing.",
            )
        with col3:
            render_metric_card(
                "Verb Ratio",
                f'{style_scores.get("verb_ratio", 0):.2f}',
                "Higher values suggest more action-driven wording.",
            )

        st.write("")
        st.write(f'Noun ratio: {style_scores.get("noun_ratio", 0):.2f}')
        st.progress(min(max(float(style_scores.get("noun_ratio", 0)), 0.0), 1.0))
        st.write(f'Adjective ratio: {style_scores.get("adj_ratio", 0):.2f}')
        st.progress(min(max(float(style_scores.get("adj_ratio", 0)), 0.0), 1.0))
        st.write(f'Adverb ratio: {style_scores.get("adv_ratio", 0):.2f}')
        st.progress(min(max(float(style_scores.get("adv_ratio", 0)), 0.0), 1.0))

    st.markdown("</div>", unsafe_allow_html=True)


def render_aspects(aspect_scores):
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Aspect Sentiment</div>', unsafe_allow_html=True)

    if not aspect_scores:
        st.info("No aspect-level analysis available.")
    else:
        cols = st.columns(len(aspect_scores))
        for column, (aspect, value) in zip(cols, aspect_scores.items()):
            with column:
                render_metric_card(
                    aspect.capitalize(),
                    str(value).capitalize(),
                    "Detected from sentences mentioning this aspect.",
                )

    st.markdown("</div>", unsafe_allow_html=True)


def render_review_block(title, body, note=None):
    note_html = f'<div class="small-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="review-label">{title}</div>
            <div class="review-body">{body}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_similar_reviews(similar_reviews):
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Similar Reviews</div>', unsafe_allow_html=True)

    if not similar_reviews:
        st.info(
            "No similar reviews available yet. Generate `bert_embeddings.pkl` and "
            "`reviews.pkl` first to enable this section."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for item in similar_reviews:
        if isinstance(item, dict):
            rank = item.get("rank", "-")
            score = item.get("similarity_score", "-")
            review_text = item.get("review", "")
            st.markdown(
                f"""
                <div class="similarity-card">
                    <div class="score-chip">Match {rank} | Score {score}</div>
                    <div class="review-body">{review_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="similarity-card">
                    <div class="review-body">{item}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def fetch_analysis(review):
    response = requests.post(API_URL, json={"review": review}, timeout=60)
    response.raise_for_status()
    return response.json()


inject_styles()
render_hero()

left_col, right_col = st.columns([1.2, 0.8], gap="large")

with left_col:
    review = st.text_area(
        "Enter Movie Review",
        value=st.session_state.get("review_input", DEFAULT_REVIEW),
        height=220,
        placeholder="Paste a movie review here...",
    )
    st.session_state["review_input"] = review

with right_col:
    st.markdown(
        """
        <div class="panel-card">
            <div class="section-title">How To Use</div>
            <div class="review-body">
                1. Paste a review or keep the sample text.<br>
                2. Click <b>Analyze Review</b>.<br>
                3. Read sentiment, emotion, style, aspect, and similarity results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


analyze = st.button("Analyze Review", type="primary", use_container_width=True)

if analyze:
    if review.strip() == "":
        st.warning("Please enter a review before running the analysis.")
    else:
        try:
            with st.spinner("Analyzing review..."):
                result = fetch_analysis(review)
        except requests.RequestException as exc:
            st.error(f"Could not connect to backend: {exc}")
        else:
            render_review_block("Original Review", result.get("review", ""))
            render_review_block(
                "Processed Review",
                result.get("cleaned_review", ""),
                note="This is the cleaned review text sent into the trained models.",
            )

            col1, col2 = st.columns([0.95, 1.05], gap="large")
            with col1:
                render_sentiment(result.get("sentiment"))
                render_emotions(result.get("emotion", {}))
            with col2:
                render_style(result.get("style", {}))

            render_aspects(result.get("aspect", {}))
            render_similar_reviews(result.get("similar_reviews", []))

            with st.expander("Raw API Response"):
                st.json(result)
