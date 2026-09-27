from pathlib import Path
import pytest
from scripts.configure_scanner_credentials import read_credentials


def test_import_accepts_only_the_two_generated_runtime_logins(tmp_path):
    path = tmp_path/'credentials.csv'
    path.write_text('role_name,password\nwatchers_scanner_api,'+'a'*64+'\nwatchers_scanner_worker,'+'b'*64+'\n')
    assert set(read_credentials(path)) == {'watchers_scanner_api','watchers_scanner_worker'}


@pytest.mark.parametrize('contents', [
    'role_name,password\npostgres,'+'a'*64+'\nwatchers_scanner_worker,'+'b'*64,
    'role_name,password\nwatchers_scanner_api,short\nwatchers_scanner_worker,'+'b'*64,
    'role_name,password\nwatchers_scanner_api,'+'a'*64+'\nwatchers_scanner_api,'+'b'*64,
])
def test_wrong_account_or_malformed_exports_cannot_be_used(tmp_path, contents):
    path = tmp_path/'credentials.csv'
    path.write_text(contents)
    with pytest.raises(ValueError):
        read_credentials(path)
