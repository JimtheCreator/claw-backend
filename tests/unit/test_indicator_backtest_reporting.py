import json
import sys

import pandas as pd

from tests.backtesting.run_indicator_evidence import main


def test_reports_can_be_rebuilt_without_rerunning_detectors(tmp_path, monkeypatch):
    for start,end,r in (("2025-06-15","2025-07-15",-1.),("2025-07-15","2025-08-15",2.)):
        row=dict(symbol="BTCUSDT",tf="5m",mtfa=False,version="frozen_v2",timestamp=start+"T12:00:00+00:00",
                 r=r,gross_r=r+0.1,vwap_side=True,volume_profile_side=False,
                 tsmom_alignment=None,cvd_break_confirmation=None,no_opposing_divergence=True)
        (tmp_path/f"checkpoint-BTCUSDT-{start}-{end}.json").write_text(json.dumps(([row],[],[])))
    monkeypatch.setattr(sys,"argv",["study","--cache",str(tmp_path),"--output",str(tmp_path),
        "--start","2025-06-15","--split","2025-07-15","--end","2025-08-15",
        "--symbols","BTCUSDT","--summarize-only"])
    main()
    assert len(pd.read_csv(tmp_path/"trades.csv"))==2
    report=pd.read_csv(tmp_path/"per-feature.csv")
    assert len(report)==30
    assert report.point_biserial_r.isna().all()
    missing=report[(report.feature=="cvd_break_confirmation") & (report.state=="unavailable")]
    assert missing.n.tolist()==[1,1]
