import json

import requests
from adtool.db.utils.config import Config

config = Config()


def _get_discoveries_with_filter(_filter, _query=None):
    _filter = "filter=" + json.dumps(_filter)
    _query = "&query=" + json.dumps(_query) if _query else ""
    return json.loads(
        requests.get(
            url="http://{}:{}/discoveries?{}{}".format(
                config.EXPEDB_CALLER_HOST, config.EXPEDB_CALLER_PORT, _filter, _query
            )
        ).content.decode()
    )
