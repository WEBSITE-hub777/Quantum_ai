import os
import sqlite3
from pathlib import Path
from typing import Any


DATABASE_PATH = Path(
    os.getenv(
        "DATABASE_PATH",
        "quantum_queen.db",
    )
)


def get_connection() -> sqlite3.Connection:

    connection = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


def initialize_database() -> None:

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            message_type TEXT DEFAULT 'normal',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(chat_id)
            REFERENCES chats(id)
            ON DELETE CASCADE
        )
        """
    )

    connection.commit()

    connection.close()


def create_chat(
    title: str = "New Chat",
) -> int:

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO chats (title)
        VALUES (?)
        """,
        (title,),
    )

    chat_id = cursor.lastrowid

    connection.commit()

    connection.close()

    return int(chat_id)


def get_chats() -> list[dict[str, Any]]:

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            title,
            created_at,
            updated_at
        FROM chats
        ORDER BY updated_at DESC
        """
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


def get_chat(
    chat_id: int,
) -> dict[str, Any] | None:

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            title,
            created_at,
            updated_at
        FROM chats
        WHERE id = ?
        """,
        (chat_id,),
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


def add_message(
    chat_id: int,
    role: str,
    content: str,
    message_type: str = "normal",
) -> int:

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO messages (
            chat_id,
            role,
            content,
            message_type
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            chat_id,
            role,
            content,
            message_type,
        ),
    )

    message_id = cursor.lastrowid

    cursor.execute(
        """
        UPDATE chats
        SET updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (chat_id,),
    )

    connection.commit()

    connection.close()

    return int(message_id)


def get_messages(
    chat_id: int,
) -> list[dict[str, Any]]:

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            chat_id,
            role,
            content,
            message_type,
            created_at
        FROM messages
        WHERE chat_id = ?
        ORDER BY id ASC
        """,
        (chat_id,),
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


def update_chat_title(
    chat_id: int,
    title: str,
) -> bool:

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE chats
        SET
            title = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (
            title,
            chat_id,
        ),
    )

    changed = cursor.rowcount > 0

    connection.commit()

    connection.close()

    return changed


def delete_chat(
    chat_id: int,
) -> bool:

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM messages
        WHERE chat_id = ?
        """,
        (chat_id,),
    )

    cursor.execute(
        """
        DELETE FROM chats
        WHERE id = ?
        """,
        (chat_id,),
    )

    deleted = cursor.rowcount > 0

    connection.commit()

    connection.close()

    return deleted


initialize_database()