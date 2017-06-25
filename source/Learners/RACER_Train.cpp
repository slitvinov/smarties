/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

void RACER::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	//this should go to gamma rather quick:
	const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
	const Real rGamma = annealedGamma();

	const Uint ndata = data->Set[seq]->tuples.size();
	assert(samp<ndata-1);
	const bool bEnd = data->Set[seq]->ended;
	const Uint nMaxTargets = MAX_UNROLL_AFTER+1;
	//for off policy correction we need reward and action, therefore not last one:
	const Uint nSUnroll = min(ndata-1-samp, nMaxTargets);
	//if we do not have a terminal reward, then we compute value of last state:
	const Uint nSValues = min(bEnd? ndata-1-samp :ndata-samp, nMaxTargets);
	//to prevent silly overflow on aux tasks:
	const Uint nSalloc = max(nSValues, static_cast<Uint>(2));

	if(thrID==1) profiler->stop_start("FWD");

	vector<vector<Real>> out_cur(1, vector<Real>(nOutputs,0));
	vector<vector<Real>> out_hat(nSValues, vector<Real>(nOutputs,0));
	vector<Activation*> series_cur = net->allocateUnrolledActivations(1);
	vector<Activation*> series_hat = net->allocateUnrolledActivations(nSalloc);
	//printf("%d %u %u %u %u %u \n", bEnd, samp, ndata, nSUnroll, nSValues, nSalloc); fflush(0);
	for (Uint k=0; k<nSValues; k++) {
		const Tuple * const _t = data->Set[seq]->tuples[k+samp]; //this tuple contains s, a, mu
		const vector<Real> inp = data->standardize(_t->s);
		//const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
		if(!k) net->seqPredict_inputs(inp, series_cur[k]);
		net->seqPredict_inputs(inp, series_hat[k]);
	}
	net->seqPredict_execute(series_cur, series_cur);
	net->seqPredict_execute(series_hat, series_hat, net->tgt_weights, net->tgt_biases);
	for (Uint k=0; k<nSValues; k++) {
		if(!k) net->seqPredict_output(out_cur[k], series_cur[k]);
		net->seqPredict_output(out_hat[k], series_hat[k]);
	}

	if(thrID==1)  profiler->stop_start("ADV");

	Real Q_RET = 0, Q_OPC = 0;
	//if partial sequence then compute value of last state (=! R_end)
	if(nSValues != nSUnroll) {
		assert(nSValues>nSUnroll && !bEnd);
		Q_RET=Q_OPC= out_hat[nSValues-1][net_indices[0]]; //V(s_T) with tgt weights
	} else assert(data->Set[seq]->tuples[ndata-1]->mu.size() == 0);

	for (int k=static_cast<int>(nSUnroll)-1; k>0; k--) //propagate Q to k=0
		offPolCorrUpdate(seq, k+samp, Q_RET, Q_OPC, out_hat[k], rGamma);

	if(thrID==1)  profiler->stop_start("CMP");

	vector<Real> grad = compute<0>(seq, samp, Q_RET, Q_OPC, out_cur[k], out_hat[k], rGamma, thrID);

	#ifdef FEAT_CONTROL
		task->Train(series_cur[k],series_hat[k+1],act,seq,samp,rGamma,gradient);
	#endif

	//write gradient onto output layer:
	clip_gradient(gradient, stdGrad[0], seq, samp);
	net->setOutputDeltas(gradient, series_cur[k]);

	if(thrID==1)  profiler->stop_start("BCK");

	if (thrID==0) net->backProp(series_cur, net->grad);
	else net->backProp(series_cur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series_cur);
	net->deallocateUnrolledActivations(&series_hat);
	if(thrID==1)  profiler->stop_start("TSK");
}
