import streamlit as st


def page_icon(emoji: str):
    """
    Shows an emoji as a Notion-style page icon.

    :param emoji: The emoji to display.

    Returns:
        None
    """

    st.write(
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

    st.write(
        f'<span class="material-symbols-outlined">{name}</span>',
        unsafe_allow_html=True,
    )
