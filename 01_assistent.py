import os
import ollama
import streamlit as st
from openai import OpenAI

from utilities.icon import (
    get_material_image,
    material_icon,
    get_bot_avatar,
    get_human_avatar,
)
from streamlit_extras.app_logo import add_logo

from utilities.rag import (
    StreamHandler,
    get_contexts,
    get_custom_rag_chain,
    set_new_context,
)

st.set_page_config(
    page_title="AI assistent",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
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
    """
    The main function that runs the application.
    """

    add_logo("assets/logo-small.png", height=150)
    # st.logo("./assets/logo.png", link="https://www.dikw.com/")

    get_material_image("chat_bubble", width=50)

    st.subheader("Chat assistent", divider="orange", anchor=False)

    if not "initialized" in st.session_state:
        st.session_state.new_context = []
        st.session_state.messages = []
        st.session_state.initialized = True

    # try to connect to ollama server if successful show available models
    # if not show a warning and a button to go to settings

    available_models = None

    try:
        models_info = ollama.list()
        available_models = extract_model_names(models_info)
    except Exception as e:
        st.warning(
            f"Could not connect to Ollama server, please check your internet connection or try again later. Error: {e}",
            icon="⚠️",
        )

    col1, col2 = st.columns(2)

    with col1:

        if available_models:
            selected_model = st.selectbox(
                "Kies een taal model dat beschikbaar is voor jou. ↓",
                available_models,
                index=1,
                help="Er zijn verschillende  grote taal modellen beschikbaar, selecteer hier een model dat voor deze toepassing geschikt is.",
            )

        else:
            st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
            if st.button("Go to settings to download a model"):
                st.page_switch("pages/03_settings.py")
    with col2:
        # show selectbox to select contexts
        contexts = get_contexts()
        selected_context = st.selectbox(
            "Selecteer een context waarmee je wilt chatten, free format is default en zonder context. ",
            contexts,
            key="context",
            on_change=set_new_context,
            help="Kies een context waarmee je wilt chatten, of kies free format voor een vrije chat. De contexten zijn gebaseerd op documenten die zijn geladen die over een speciefiek onderwerp gaan.",
        )

    # setup message container
    message_container = st.container(height=500, border=True)

    # select chat client based on context
    if selected_context == "free format":
        client = OpenAI(
            base_url="http://pcloud:11434/v1",
            api_key="ollama",  # required, but unused
        )
    else:
        # setup rag chain
        qa_client = get_custom_rag_chain(selected_context)

    # if contexted changed
    if st.session_state.new_context:
        # clear the chat history
        st.session_state.messages = []
        # but inform the user we switched context
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"You switched context to {st.session_state.context}",
            }
        )

        st.session_state.new_context = False

    for message in st.session_state.messages:
        avatar = (
            get_bot_avatar() if message["role"] == "assistant" else get_human_avatar()
        )
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a prompt here..."):
        try:
            st.session_state.messages.append({"role": "user", "content": prompt})

            message_container.chat_message(
                "user",
                avatar=get_human_avatar(),
            ).markdown(prompt)

            with message_container.chat_message(
                "assistant",
                avatar=get_bot_avatar(),
            ):
                # use different clients based on context
                if selected_context == "free format":
                    with st.spinner("model working..."):
                        stream = client.chat.completions.create(
                            model=selected_model,
                            messages=[
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ],
                            stream=True,
                        )
                    # stream response
                    response = st.write_stream(stream)

                else:
                    with st.spinner("model working..."):
                        # TODO use invoke or stream
                        stream = qa_client.stream(
                            {"question": prompt}
                        )  # , callback=callback)
                        # print(response)
                    response = st.write_stream(stream)  # or st.write_stream(stream)

            # container
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(e, icon="⛔️")


if __name__ == "__main__":
    main()
