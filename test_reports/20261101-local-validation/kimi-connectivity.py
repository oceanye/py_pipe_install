"""Read-only official Kimi model-list probe; never saves credentials or photos.

Reads a credential from stdin. Keep stdin out of terminal echo/history. No
provider body, headers, account data or credentials are written to evidence.
This is not the application's missing AI adapter or a vision acceptance test.
"""
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
import warnings

ENDPOINT = 'https://api.moonshot.cn/v1/models'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    if '--interactive' in sys.argv:
        warnings.simplefilter('error', getpass.GetPassWarning)
        credential = getpass.getpass('Kimi API key (hidden): ').strip()
    else:
        print('Waiting for credential on non-echoed stdin.', flush=True)
        credential = sys.stdin.readline().strip()
    if not credential:
        raise SystemExit('No credential received; no request sent.')
    request = urllib.request.Request(ENDPOINT, headers={
        'Authorization': 'Bearer ' + credential, 'Accept': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    evidence = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'endpoint': ENDPOINT, 'method': 'GET',
        'scope': 'OFFICIAL_PROVIDER_MODEL_LIST_ONLY_NOT_APP_INTEGRATION',
        'credential_saved': False, 'image_uploaded': False,
        'inference_requested': False, 'application_ai_wiring_validated': False,
        'documentation': 'https://platform.kimi.com/docs/api/list-models',
    }
    try:
        with opener.open(request, timeout=20) as response:
            evidence['http_status'] = response.status
            payload = json.loads(response.read(1024 * 1024))
        models = [row.get('id', '') for row in payload.get('data', []) if isinstance(row, dict)]
        evidence['model_ids'] = [m for m in models
                                 if isinstance(m, str) and re.fullmatch(r'[A-Za-z0-9._/-]{1,120}', m)]
        evidence['model_list_returned'] = bool(evidence['model_ids'])
        evidence['status'] = 'MODEL_LIST_RECEIVED'
    except urllib.error.HTTPError as error:
        evidence.update(status='HTTP_ERROR', http_status=error.code)
    except Exception as error:
        # Exception messages and provider bodies may echo sensitive input.
        evidence.update(status='CONNECTION_OR_PARSE_ERROR', error_type=type(error).__name__)
    finally:
        credential = ''
        request.remove_header('Authorization')
    try:
        with opener.open(ENDPOINT, timeout=20) as response:
            evidence['unauthenticated_http_status'] = response.status
    except urllib.error.HTTPError as error:
        evidence['unauthenticated_http_status'] = error.code
    except Exception as error:
        evidence['unauthenticated_error_type'] = type(error).__name__
    evidence['model_list_authentication_verified'] = (
        evidence.get('http_status') == 200 and
        evidence.get('unauthenticated_http_status') in (401, 403))
    Path(__file__).with_name('kimi-connectivity.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
