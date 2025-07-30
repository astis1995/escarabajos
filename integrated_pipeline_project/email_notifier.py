# email_notifier.py

import smtplib
from email.message import EmailMessage
import ssl

# Configuration (edit these before use)
GMAIL_ADDRESS = "your_email@gmail.com"
GMAIL_APP_PASSWORD = "your_app_password"
TO_EMAIL = "recipient@example.com"

def send_email(message: str, subject: str = "Notification from Python"):
    """
    Sends an email with the given message and subject using Gmail.

    Parameters:
    - message (str): The body of the email.
    - subject (str): Optional subject line. Default: "Notification from Python"
    """
    email = EmailMessage()
    email["From"] = GMAIL_ADDRESS
    email["To"] = TO_EMAIL
    email["Subject"] = subject
    email.set_content(message)

    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp.send_message(email)
            print(f"[EMAIL] ✅ Email sent to {TO_EMAIL}")
    except Exception as e:
        print(f"[EMAIL] ❌ Failed to send email: {e}")
