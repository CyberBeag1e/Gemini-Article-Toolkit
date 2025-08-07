import random
from src.pipeline import NewsProcessor
import streamlit as st

class App:

    def __init__(self):
        # Suppored default colors in streamlit
        self.palette = ["blue", "green", "orange", "red", "violet", "gray", "primary"]
    
    def main_view(self):
        inp, res = st.columns(2, border = True)

        with inp:
            uploaded = st.file_uploader("Upload a .txt file", type = ["txt"])
            if uploaded:
                uploaded_text = uploaded.read().decode("utf-8")
                article = st.text_area(
                    label = "Article",
                    height = 300,
                    value = uploaded_text,
                    disabled = True
                )
            
            else:
                article = st.text_area(
                    label = "Or paste your article",
                    height = 300,
                    value = "",
                    placeholder = "Paste or type the article here."
                )

            entity = st.text_input(
                label = "Entity name (optional)",
                value = "",
                placeholder = "Enter an entity name, e.g. a person, a company, a character, etc."
            )

            question = st.text_input(
                label = "Question (optional)",
                value = "",
                placeholder = "Enter a question about the article."
            )

            button = st.button(
                label = "Run",
                disabled = not article.strip()
            )
        
        with res:
            if button:
                try:
                    processor = NewsProcessor(article = article, 
                                              entity = entity or None)
                    
                    with st.spinner("Summarizing..."):
                        summary = processor.summarize()
                    
                    st.subheader("Summary")
                    st.markdown(f"##### {summary["headline"]}")
                    st.write(f"**Summary**: {summary["summary"]}")

                    # Convert each topic into a streamlit badge (st.badge)
                    colored_topics = zip(summary['topics'], self.palette)
                    st.write(f"""
                    **Topics**: {' '.join([f':{clr}-badge[{tp}]' for tp, clr in colored_topics])}
                    """)

                    if entity:
                        with st.spinner("Finding references..."):
                            references = processor.highlight_entity()
                    
                        st.subheader(f"Facts about :primary-badge[{entity}]")
                        st.write(
                            "\n".join(f"- {s}" for s in references["sent_lst"])
                            or "_No references found_"
                        )
                    
                    if question:
                        with st.spinner("Answering question..."):
                            answer = processor.answer_question(question)
                        
                        st.subheader("Q&A")

                        if answer:
                            st.markdown(f"**Answer**: {answer['answer']}")

                            # Convert each evidence keyword into a streamlit badge (st.badge).
                            if answer["evidence"]:
                                for ev in answer["evidence"]:
                                    color = random.choice(self.palette)
                                    article = article.replace(ev, f":{color}-badge[{ev}]")
                            
                            st.write(f"**Evidence**: {article}")
                        
                        else:
                            st.warning("Model did not return a usable answer.")
                
                except Exception as e:
                    st.error(f"Something went wrong, please try again: {e}")
    
    def run(self):
        st.set_page_config(page_title = "Gemini Article Toolkit", layout = "wide")
        st.title("Gemini Article Toolkit")
        self.main_view()
        st.caption("Developed by CyberBeag1e | Powered by Streamlit, Google Gemini")
    

if __name__ == "__main__":
    app = App()
    app.run()
