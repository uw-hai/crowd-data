# crowd-data
Crowdsourcing Datasets.

`crowddata.answers` contains wrappers for datasets with worker answers.
`crowddata.answers.Data.from_lin_aaai12()` loads datasets described in [1], and referred to as LinWiki and LinTag in [2].
`crowddata.questions` contains resources relating to questions for workers. (Incomplete)

Data available on request for research purposes.

## Dataset creation for [3]

The Travel dataset in [3] was created by executing:
```bash
python -c 'import crowddata.questions.dmoz_data as dd; dd.make_travel_dataset()'
python -c "import os; import crowddata.questions.util as ut; ut.partition(os.path.join(os.environ['DMOZ_TRAVEL_WEB'], 'data.json')), [300, 100, -1], priorities=[0, 1, 1], seed=0)"
```

The Cars dataset in [3] was created by executing:
```bash
python -c 'import crowddata.questions.imagenet_data as ID; ID.make_car_or_not_dataset()'
python -c "import crowddata.questions.util as ut; ut.partition(os.path.join(os.environ['CAR_OR_NOT_WEB']/data.json', [300, 100, -1], priorities=[0, 1, 1], seed=0)"
```


## References
- [1] Christopher H. Lin, Mausam, and Daniel S. Weld. 2012. [Dynamically Switching between Synergistic Workflows for Crowdsourcing](https://homes.cs.washington.edu/~chrislin/papers/aaai12.pdf). In Proceedings of the 26th AAAI Conference on Artificial Intelligence (AAAI).
- [2] Jonathan Bragg, Mausam, and Daniel S. Weld. 2016. [Optimal Testing for Crowd Workers](https://www.cs.washington.edu/ai/pubs/bragg-aamas16.pdf). In Autonomous Agents and Multiagent Systems (AAMAS '16).
- [3] Jonathan Bragg, Mausam, and Daniel S. Weld. 2018. [Sprout: Crowd-Powered Task Design for Crowdsourcing](https://cs.stanford.edu/~jbragg/files/bragg-uist18.pdf). In ACM Symposium on User Interface Software and Technology (UIST'18).
