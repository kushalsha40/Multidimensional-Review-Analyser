import requests
import streamlit as st


st.title("IMDB Review Analysis")
st.write("Analyze movie reviews using our NLP-powered analysis tool!")


def render_similar_reviews(similar_reviews):
    st.subheader("Similar Reviews")

    if not similar_reviews:
        st.info(
            "No similar reviews available yet. Generate `bert_embeddings.pkl` and "
            "`reviews.pkl` first to enable this section."
        )
        return

    for item in similar_reviews:
        if isinstance(item, dict):
            rank = item.get("rank", "-")
            score = item.get("similarity_score", "-")
            review_text = item.get("review", "")
            st.markdown(f"**Match {rank}**  |  Similarity score: `{score}`")
            st.write(review_text)
            st.divider()
        else:
            # Backward compatibility if the API still returns a plain list of strings.
            st.write(item)
            st.divider()


review = st.text_area("Enter Movie Review")


if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        url = "http://localhost:8000/analyze"
        payload = {"review": review}

        try:
            response = requests.post(url, json=payload, timeout=60)
        except requests.RequestException as exc:
            st.error(f"Could not connect to backend: {exc}")
        else:
            if response.status_code == 200:
                result = response.json()

                st.write("Full API Response:")
                st.write(result)

                st.subheader("Sentiment")
                st.write(result.get("sentiment", "Not found"))

                st.subheader("Emotion Score")
                st.json(result.get("emotion", {}))

                st.subheader("Writing Style")
                st.json(result.get("style", {}))

                st.subheader("Aspect Sentiment")
                st.json(result.get("aspect", {}))

                render_similar_reviews(result.get("similar_reviews", []))
            else:
                st.error(f"API request failed: {response.status_code}")
                st.write(response.text)
