import os
import ollama
import streamlit as st
from openai import OpenAI

from utilities.icon import material_icon, get_bot_avatar, get_human_avatar
from streamlit_extras.app_logo import add_logo

from utilities.rag import StreamHandler, get_custom_rag_chain

st.set_page_config(
    page_title="AI assistent",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

add_logo("assets/logo-small.png", height=100)


def extract_model_names(models_info: list) -> tuple:
    """
    Extracts the model names from the models information.

    :param models_info: A dictionary containing the models' information.

    Return:
        A tuple containing the model names.
    """

    return tuple(model["name"] for model in models_info["models"])


def get_contexts() -> tuple:
    """
    Returns the contexts available in the system.

    :return: A tuple containing the contexts.
    """
    # we start with a default "free format" context plus we add  all contexts available in the context folder
    contexts = ["free format"]
    # get all sub directory names of the directotry contexts
    context_folders = os.listdir("./contexts")
    # add all sub directory names to the contexts list
    contexts.extend(context_folders)

    return tuple(contexts)


def set_new_context():
    """
    Clears the chat history.
    """
    # set new context to True
    st.session_state.new_context = True
    return


def main():
    """
    The main function that runs the application.
    """

    material_icon("chat_bubble")

    st.subheader("Chat assistent", divider="orange", anchor=False)

    if not "initialized" in st.session_state:
        st.session_state.new_context = []
        st.session_state.messages = []
        st.session_state.initialized = True

    col1, col2 = st.columns(2)

    with col1:

        models_info = ollama.list()
        available_models = extract_model_names(models_info)

        if available_models:
            selected_model = st.selectbox(
                "Pick a model available on your system ↓", available_models
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
                    # use the custom rag chain
                    # setup the callback
                    # callback = StreamHandler(st.empty())
                    with st.spinner("model working..."):
                        stream = qa_client({"question": prompt})  # , callback=callback)
                        # print(response)
                    response = st.write(stream["answer"])
                    # answer = response["answer"]
                    # message_container.chat_message(
                    #     "assistant",
                    #     avatar=get_bot_avatar(),
                    # ).markdown(answer)
            # container
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            st.error(e, icon="⛔️")


if __name__ == "__main__":
    main()
