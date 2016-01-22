========
Usage
========

To use brew in a project::

	import brew
	from brew.base import Ensemble
	from brew.base import EnsembleClassifier
	from brew.combination import Combiner
	
	# here, clf1 and clf2 are sklearn classifiers or brew ensemble classifiers
	# already trained. Keep in mind that brew requires your labels = [0,1,2,...]
	# numerical with no skips.
	clfs = [clf1, clf2]
	ens = Ensemble(classifiers = clfs)
	
	# create your Combiner
	# the rules can be 'majority_vote', 'max', 'min', 'mean' or 'median'
	comb = Combiner(rule='majority_vote')
	
	# now create your ensemble classifier
	ensemble_clf = EnsembleClassifier(ensemble=ens, combiner=comb)
	y_pred = ensemble_clf.predict(X)
	
	# there you go, y_pred is your prediction.
