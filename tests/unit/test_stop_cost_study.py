import pandas as pd

from tests.backtesting.run_stop_cost_study import freeze


def test_dev_selection_is_frozen_by_predeclared_rule(tmp_path):
    rows=[]
    for floor,n,mean in ((0,100,-1.0),(12,90,-.9),(18,60,-.7),(24,40,-.5),(30,20,.2),(40,35,-.5),(50,5,2.0)):
        rows.append(dict(planner="old",min_stop_bps=floor,n=n,avg_net_r=mean))
    rows.append(dict(planner="new",min_stop_bps=12,n=999,avg_net_r=99))
    policy=freeze(pd.DataFrame(rows),tmp_path)
    # 30/50 fail N/retention. 24 and 40 tie on mean; 24 wins on N.
    assert policy["threshold_bps"]==24
    assert policy["selection_source"]=="old planner development only"


def test_no_filter_is_frozen_when_dev_improvement_is_too_small(tmp_path):
    frame=pd.DataFrame([
        dict(planner="old",min_stop_bps=0,n=100,avg_net_r=-1.0),
        dict(planner="old",min_stop_bps=24,n=60,avg_net_r=-.8),
    ])
    policy=freeze(frame,tmp_path)
    assert policy["threshold_bps"]==0 and not policy["development_candidate"]
    assert policy["production_default"] is False
