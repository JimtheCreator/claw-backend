"""Verified PostgreSQL TLS, including Supabase's published legacy root CA."""
import hashlib
import ssl
from pathlib import Path


# Supabase dashboard's prod-ca-2021.crt lacks a CA keyUsage extension.
# Python 3.13 enables X509_STRICT by default; the official legacy root needs
# pre-3.13 extension handling. Only this exact certificate gets that exception.
SUPABASE_LEGACY_ROOT_SHA256 = '807025ad50d4ed219d2c9c7d299c004f824eb00cf7f65afef607d07b72e6cafa'


def database_tls_context(cafile=None):
    context = ssl.create_default_context(cafile=cafile)
    if cafile:
        pem = Path(cafile).read_text(encoding='ascii')
        if pem.count('-----BEGIN CERTIFICATE-----') == 1:
            der = ssl.PEM_cert_to_DER_cert(pem)
            if hashlib.sha256(der).hexdigest() == SUPABASE_LEGACY_ROOT_SHA256:
                context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    # Chain trust, expiry and hostname verification remain mandatory.
    return context
