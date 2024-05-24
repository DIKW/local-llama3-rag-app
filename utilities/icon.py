import streamlit as st

from PIL import Image


def page_icon(emoji: str):
    """
    Shows an emoji as a Notion-style page icon.

    :param emoji: The emoji to display.

    Returns:
        None
    """

    return st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


def material_icon(name: str):
    """
    Shows a Material icon.

    :param name: The name of the Material icon.

    Returns:
        None
    """

    return st.write(
        f'<span class="material-symbols-outlined">{name}</span>',
        unsafe_allow_html=True,
    )


def get_material_image(name: str, width: int = None):
    """
    Shows a Material icon.

    :param name: The name of the Material icon.

    Returns:
        None
    """
    image = Image.open(f"./assets/{name}.png")

    return st.image(image, width=width)


def get_bot_avatar():
    """
    Returns the avatar of the bot.

    :return: The avatar of the bot.
    """

    return "https://raw.githubusercontent.com/DIKW/local-llama3-rag-app/main/assets/dikw-chatbot-2.png"


def get_human_avatar():
    """
    Returns the avatar of the human.

    :return: The avatar of the human.
    """

    return "https://raw.githubusercontent.com/DIKW/local-llama3-rag-app/main/assets/person.png"
