"""
Module to manage chat history using a SQL database. Currently implemented using
a local SQLite database, but could be extended to use other SQL databases (e.g.
PostgreSQL, MySQL) in the future for production use.
"""

from langchain_core.messages import BaseMessage
from langchain_community.chat_message_histories.sql import (
    SQLChatMessageHistory,
)

from sqlalchemy import create_engine


class ChatHistory:
    """
    Class to manage chat history using a SQL database, ith methods to get and
    add messages to the database, along with clearing the history for a
    specific session.
    """

    def __init__(self) -> None:
        self._engine = create_engine(url="sqlite:///chat_history.db")

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        """
        Get messages from the database for a specific session.

        :param session_id: The ID of the session to get messages for.
        :type session_id: str
        :return: A list of messages for the specified session.
        :rtype: list[BaseMessage]
        """
        return SQLChatMessageHistory(
            connection=self._engine,
            session_id=session_id,
        ).get_messages()

    def add_messages(
        self, session_id: str, messages: list[BaseMessage]
    ) -> None:
        """
        Add messages to the database for a specific session.

        :param session_id: The ID of the session to add messages for.
        :type session_id: str
        :param messages: The list of messages to add.
        :type messages: list[BaseMessage]
        """
        SQLChatMessageHistory(
            connection=self._engine,
            session_id=session_id,
        ).add_messages(messages=messages)

    def clear(self, session_id: str) -> None:
        """
        Clear the chat history for a specific session.

        :param session_id: The ID of the session to clear.
        :type session_id: str
        """
        SQLChatMessageHistory(
            connection=self._engine,
            session_id=session_id,
        ).clear()
