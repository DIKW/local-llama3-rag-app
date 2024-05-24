import streamlit as st
import json
import ollama
from utilities.icon import (
    get_bot_avatar,
    get_human_avatar,
    get_material_image,
    material_icon,
)

from streamlit_extras.app_logo import add_logo

from utilities.rag import get_contexts, set_new_context
from utilities.rag import get_vector_db


st.set_page_config(
    page_title="Semantic Search Playground",
    page_icon=":material/forum:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_allowed_model_names(models_info: dict) -> tuple:
    """
    Returns a tuple containing the names of the allowed models.
    """
    allowed_models = ["llama3:latest", "llava:latest"]
    return tuple(
        model
        for model in allowed_models
        if model in [m["name"] for m in models_info["models"]]
    )


def extract_model_names(models_info: list) -> tuple:
    """
    Extracts the model names from the models information.

    :param models_info: A dictionary containing the models' information.

    Return:
        A tuple containing the model names.
    """

    return tuple(model["name"] for model in models_info["models"])


def main():
    add_logo("assets/logo-small.png", height=150)

    get_material_image("search", width=50)

    st.subheader("Semantisch zoeken", divider="orange", anchor=False)

    st.info(
        """Dit is een demo van semantisch zoeken met behulp van de RAG-assistent. 
Stel een vraag over de documenten in de context en ik zal de meest relevante documenten voor je vinden. \n
We zoeken in de database stukjes tekst uit de documenten die 'dicht bij' de vraag liggen die je stelt. 
Stukjes tekst met een **lage score** liggen dichter bij je vraag dan stukjes met een **hoge score**."""
    )

    col_1, col_2, col_3 = st.columns(3)

    with col_1:
        # set the number of documents to return
        k = st.number_input(
            "Geef het aantal documenten op dat je terug wilt hebben",
            min_value=1,
            max_value=100,
            value=3,
            step=1,
            key="n_docs",
        )
    with col_2:
        # set the snippet length to show to user
        l = st.number_input(
            "Aantal characters dat je wilt zien van de documenten",
            min_value=10,
            max_value=250,
            value=100,
            step=10,
            key="snippet_length",
        )
    with col_3:
        # show selectbox to select contexts
        contexts = get_contexts()
        # the default 'free_format' context should be removed from the set
        contexts = [context for context in contexts if context != "free format"]

        selected_context = st.selectbox(
            "Selecteer een context waarmee je wilt chatten.",
            contexts,
            key="context",
            on_change=set_new_context,
        )

    # return the retriever
    if selected_context is not None:
        db = get_vector_db(selected_context)

    if "chats" not in st.session_state:
        st.session_state.chats = []

    # col1, col2 = st.columns(2)

    # with col1:
    container2 = st.container(height=400, border=True)

    if selected_context is not None:
        for message in st.session_state.chats:
            avatar = (
                get_bot_avatar()
                if message["role"] == "assistant"
                else get_human_avatar()
            )
            with container2.chat_message(message["role"], avatar=avatar):
                if message["role"] == "user":
                    st.markdown(message["content"])
                else:
                    st.markdown(message["content"])

        if user_input := st.chat_input(
            "Stel een vraag over de documenten in de context...", key="chat_input"
        ):
            st.session_state.chats.append({"role": "user", "content": user_input})
            container2.chat_message("user", avatar=get_human_avatar()).markdown(
                user_input
            )

            with container2.chat_message("assistant", avatar=get_bot_avatar()):
                with st.spinner(":blue[processing...]"):
                    response = db.similarity_search_with_score(
                        user_input, k=st.session_state.n_docs, fetch_k=100
                    )
                    "De database geeft de volgende resultaten"
            # returns a set of relevant documents
            l = st.session_state.snippet_length
            for r in response:
                doc = r[0]
                score = r[1]
                msg = f"""Met score {score:.1f} op pagina {doc.metadata['page']} uit {doc.metadata['source']} vind ik dit : {doc.page_content[0:l]} ...  """
                container2.chat_message("assistant", avatar=get_bot_avatar()).markdown(
                    msg, help=doc.page_content
                )

                # append to chat history
                st.session_state.chats.append(
                    {
                        "role": "assistant",
                        "content": msg,
                    }
                )

    # with col2:
    #     container1 = st.container(height=500, border=True)
    #     with container1:
    #         # show document snippets
    #         st.info("Document snippets go here")


if __name__ == "__main__":
    main()
