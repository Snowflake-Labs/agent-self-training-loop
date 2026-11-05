# /// script
# requires-python = ">=3.11"
# dependencies = ["snowflake-connector-python>=3.12", "cryptography>=42"]
# ///
"""Mint a short-lived CT_QUICKSTART_USER-restricted PAT for the gateway lab.

PAT sessions cannot create PATs (099413), so a temporary key pair is attached to
RSA_PUBLIC_KEY_2 for this step only and always removed afterwards. The token is
stored in the Snowflake secret CT_QUICKSTART_DB.AGT.AGT_PAT and in a 0600 local
file for local smoke tests. It is never printed.
"""
import os
import subprocess
import tempfile

import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

TOKEN_NAME = "AGT_LAB_PAT"
DAYS = 3
EXPECTED = ("SFDEVREL", "SFDEVREL_ENTERPRISE")
LOCAL_PATH = os.path.expanduser("~/.snowflake/agt_lab/pat")


def admin_connection():
    return snowflake.connector.connect(connection_name="DEVREL_ENTERPRISE")


key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
public_b64 = "".join(
    key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode().splitlines()[1:-1]
)

with admin_connection() as conn, conn.cursor() as cur:
    cur.execute("SELECT CURRENT_ORGANIZATION_NAME(), CURRENT_ACCOUNT_NAME(), CURRENT_USER()")
    org, account, user = cur.fetchone()
    if (org, account) != EXPECTED:
        raise SystemExit(f"Refusing to run against {org}.{account}")
    cur.execute(f"DESC USER {user}")
    props = {r[0]: r[1] for r in cur.fetchall()}
    if props.get("RSA_PUBLIC_KEY_2") not in (None, "null", ""):
        raise SystemExit("RSA_PUBLIC_KEY_2 is already in use; refusing to overwrite it.")
    cur.execute(f"ALTER USER {user} SET RSA_PUBLIC_KEY_2 = '{public_b64}'")

try:
    with tempfile.TemporaryDirectory(prefix="agt-kp-") as tmp:
        os.chmod(tmp, 0o700)
        key_path = os.path.join(tmp, "k.p8")
        with open(key_path, "wb") as fh:
            fh.write(key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ))
        os.chmod(key_path, 0o600)
        try:
            with snowflake.connector.connect(
                account="SFDEVREL-ENTERPRISE", user=user, role="ACCOUNTADMIN",
                authenticator="SNOWFLAKE_JWT", private_key_file=key_path,
            ) as kp, kp.cursor() as cur:
                cur.execute("SELECT CURRENT_ORGANIZATION_NAME(), CURRENT_ACCOUNT_NAME()")
                if tuple(cur.fetchone()) != EXPECTED:
                    raise SystemExit("Key-pair session reached the wrong account.")
                cur.execute(f"SHOW USER PROGRAMMATIC ACCESS TOKENS FOR USER {user}")
                names = [dict(zip([c[0].lower() for c in cur.description], r))["name"] for r in cur.fetchall()]
                if TOKEN_NAME in names:
                    cur.execute(f"ALTER USER {user} REMOVE PROGRAMMATIC ACCESS TOKEN {TOKEN_NAME}")
                cur.execute(
                    f"ALTER USER {user} ADD PROGRAMMATIC ACCESS TOKEN {TOKEN_NAME} "
                    f"ROLE_RESTRICTION = 'CT_QUICKSTART_USER' DAYS_TO_EXPIRY = {DAYS} "
                    "COMMENT = 'gateway-to-training lab; remove after cleanup'"
                )
                row = dict(zip([c[0].lower() for c in cur.description], cur.fetchone()))
                token = row.pop("token_secret")
                if not token or "'" in token or "\\" in token:
                    raise SystemExit("Unexpected token format; secret not written.")
                cur.execute("USE ROLE CT_QUICKSTART_USER")
                cur.execute(
                    "CREATE OR REPLACE SECRET CT_QUICKSTART_DB.AGT.AGT_PAT TYPE = GENERIC_STRING "
                    f"SECRET_STRING = '{token}'"
                )
                os.makedirs(os.path.dirname(LOCAL_PATH), mode=0o700, exist_ok=True)
                fd = os.open(LOCAL_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                os.write(fd, token.encode())
                os.close(fd)
                del token
        finally:
            subprocess.run(["rm", "-P", key_path], check=False)
finally:
    with admin_connection() as conn, conn.cursor() as cur:
        cur.execute(f"ALTER USER {user} UNSET RSA_PUBLIC_KEY_2")
        cur.execute(f"DESC USER {user}")
        slot = {r[0]: r[1] for r in cur.fetchall()}.get("RSA_PUBLIC_KEY_2")
    print({"rsa_public_key_2_after_cleanup": slot})

print({"token_name": TOKEN_NAME, "role_restriction": "CT_QUICKSTART_USER", "days": DAYS,
       "secret": "CT_QUICKSTART_DB.AGT.AGT_PAT", "local_file_mode": oct(os.stat(LOCAL_PATH).st_mode & 0o777)})
