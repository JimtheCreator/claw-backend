from pathlib import Path
import ssl

from infrastructure.database.supabase.tls import database_tls_context


def test_published_supabase_root_keeps_chain_and_hostname_verification():
    ca = Path(__file__).resolve().parents[2] / 'config/certificates/supabase-root.crt'
    context = database_tls_context(ca)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
    assert not context.verify_flags & ssl.VERIFY_X509_STRICT


def test_default_trust_policy_is_not_relaxed():
    standard = ssl.create_default_context()
    context = database_tls_context()
    assert context.verify_flags == standard.verify_flags
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
