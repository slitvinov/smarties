/*
 *  Layers.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include "Layers.h"
#include <cassert>

using namespace ErrorHandling;

/* propagate(Mem/Lab * M, Lab * N, Real* weights, Real* biases)
 *
 */

void NormalLayer::propagate(Activation* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lI = input_links;
    Real in(0);
    
    for (int n=0; n<nNeurons; n++) {
        in = 0.; //zero the first
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            in += *(outvals +lI->iI +i) * *(weights +lI->iW +n*lI->nI +i);
        }
        //}
        
        in += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = in; //first element of in
        *(N->outvals +n1stNeuron +n) = func->eval( in );
    }
}

void NormalLayer::propagate(const Activation* const M, Activation* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lR = recurrent_links;
    const Link* const lI = input_links;
    
    Real in(0);
    
    for (int n=0; n<nNeurons; n++) {
        in = 0.;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            in += *(outvals    +lI->iI +i) * *(weights +lI->iW +n*lI->nI +i);
        }
        //}
        for (int i=0; i<lR->nI; i++) {
            in += *(M->outvals +lR->iI +i) * *(weights +lR->iW +n*lR->nI +i);
        }

        in += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = in;
        *(N->outvals +n1stNeuron +n) = func->eval( in );
    }
}

void LSTMLayer::propagate(Activation* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lI = input_links;

    Real tC(0), tI(0), tF(0), tO(0);

    for (int n=0; n<nNeurons; n++) {
        tC=0.; tI=0.; tF=0.; tO=0.;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(outvals +lI->iI +i);
            tC += oVal * *(weights +lI->iW  +n*lI->nI +i);
            tI += oVal * *(weights +lI->iWI +n*lI->nI +i);
            tF += oVal * *(weights +lI->iWF +n*lI->nI +i);
            tO += oVal * *(weights +lI->iWO +n*lI->nI +i);
        }
        //}
        
        tC += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = tC;
        tI += *(biases +n1stBiasIG +n);
        *(N->iIGates +n1stCell   +n) = tI;
        tF += *(biases +n1stBiasFG +n);
        *(N->iFGates +n1stCell   +n) = tF;
        tO += *(biases +n1stBiasOG +n);
        *(N->iOGates +n1stCell   +n) = tO;
        
        const Real oC = ifun->eval(tC);
        *(N->oMCell  +n1stCell +n) = oC;
        const Real oI = sigm->eval(tI);
        *(N->oIGates +n1stCell +n) = oI;
        const Real oF = sigm->eval(tF);
        *(N->oFGates +n1stCell +n) = oF;
        const Real oO = sigm->eval(tO);
        *(N->oOGates +n1stCell +n) = oO;
        const Real oS = oC * oI;
        *(N->ostates +n1stCell +n) = oS;
        *(N->outvals +n1stNeuron +n) = func->eval(oS) * oO;
    }
}

void LSTMLayer::propagate(const Activation* const M, Activation* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;

    Real tC(0), tI(0), tF(0), tO(0);

    for (int n=0; n<nNeurons; n++) {
        tC=0.; tI=0.; tF=0.; tO=0.;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(outvals +lI->iI +i);
            tC += oVal * *(weights +lI->iW  +n*lI->nI +i);
            tI += oVal * *(weights +lI->iWI +n*lI->nI +i);
            tF += oVal * *(weights +lI->iWF +n*lI->nI +i);
            tO += oVal * *(weights +lI->iWO +n*lI->nI +i);
        }
        //}
        for (int i=0; i<lR->nI; i++) {
            const Real oVal = *(M->outvals +lR->iI +i);
            tC += oVal * *(weights +lR->iW  +n*lR->nI +i);
            tI += oVal * *(weights +lR->iWI +n*lR->nI +i);
            tF += oVal * *(weights +lR->iWF +n*lR->nI +i);
            tO += oVal * *(weights +lR->iWO +n*lR->nI +i);
        }
        
        
        tC += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = tC;
        tI += *(biases +n1stBiasIG +n);
        *(N->iIGates +n1stCell   +n) = tI;
        tF += *(biases +n1stBiasFG +n);
        *(N->iFGates +n1stCell   +n) = tF;
        tO += *(biases +n1stBiasOG +n);
        *(N->iOGates +n1stCell   +n) = tO;
        
        const Real oC = ifun->eval(tC);
        *(N->oMCell  +n1stCell +n) = oC;
        const Real oI = sigm->eval(tI);
        *(N->oIGates +n1stCell +n) = oI;
        const Real oF = sigm->eval(tF);
        *(N->oFGates +n1stCell +n) = oF;
        const Real oO = sigm->eval(tO);
        *(N->oOGates +n1stCell +n) = oO;
        const Real oS = oC * oI + *(M->ostates +n1stCell +n) * oF;
        *(N->ostates +n1stCell +n) = oS;
        *(N->outvals +n1stNeuron +n) = func->eval(oS) * oO;
    }
}
/* backPropagateDelta(Activation * prev, Activation * curr, Activation * next, Real* weights, Real* biases)
 *
 */

//given errors in output-side layer, propagate errors to current layer
inline Real NormalLayer::propagateErrors(const Link* const l, const Activation* const lab, const int iNeuron, const Real* const weights) const
{
    Real err(0.);
    if (l->LSTM) { //is this link to LSTM?
        for (int i=0; i<l->nO; i++)
            err += *(lab->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +iNeuron) +
                   *(lab->errvals +l->iO +i) * (
                   *(lab->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +iNeuron) +
                   *(lab->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +iNeuron) +
                   *(lab->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +iNeuron) );
    } else {
        for (int i=0; i<l->nO; i++)
            err += *(lab->errvals +l->iO +i) * *(weights +l->iW  +i*l->nI +iNeuron);
    }
    return err;
}

void NormalLayer::backPropagateDelta(Activation* const C, const Real* const weights, const Real* const biases) const
{
    const Link* const lO = output_links;
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        
        //for (int k=0; k<output_links->size(); k++) {
        //    const Link* const lO = (*output_links)[k];
        err += propagateErrors(lO, C, n, weights);
        //}
        
        *(C->errvals +n1stNeuron +n) = err * func->evalDiff(*(C->in_vals +n1stNeuron+n));
    }
}

void NormalLayer::backPropagateDeltaFirst(Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const
{
    const Link* const lO = output_links;
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        
        //for (int k=0; k<output_links->size(); k++) {
        //    const Link* const lO = (*output_links)[k];
        err += propagateErrors(lO, C, n, weights);
        //}
        err += propagateErrors(recurrent_links, N, n, weights);
        
        *(C->errvals +n1stNeuron +n) = err * func->evalDiff(*(C->in_vals +n1stNeuron+n));
    }
}

void LSTMLayer::backPropagateDeltaFirst(Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const
{
    const Link* const lO = output_links;
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        
        //for (int k=0; k<output_links->size(); k++) {
        //    const Link* const lO = (*output_links)[k];
        err += propagateErrors(lO, C, n, weights);
        //}
        err += propagateErrors(recurrent_links, N, n, weights);

        *(C->eMCell +n1stCell+n) =       ifun->evalDiff(*(C->in_vals+n1stNeuron+n)) * *(C->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) =       sigm->evalDiff(*(C->iIGates+n1stCell  +n)) * *(C->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = 0.0;
        *(C->eOGates+n1stCell+n) = err * sigm->evalDiff(*(C->iOGates+n1stCell  +n)) * func->eval(*(C->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = err * *(C->oOGates+n1stCell+n) * func->evalDiff(*(C->ostates +n1stCell +n)) + *(N->errvals+n1stNeuron+n)* *(N->oFGates+n1stCell+n);
    }
}

void LSTMLayer::backPropagateDelta(Activation* const C, const Real* const weights, const Real* const biases) const
{
    const Link* const lO = output_links;
    for (int n=0; n<nNeurons; n++)
    {
        Real err = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        
        //for (int k=0; k<output_links->size(); k++) {
        //    const Link* const lO = (*output_links)[k];
        err += propagateErrors(lO, C, n, weights);
        //}

        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(C->in_vals+n1stNeuron+n)) * *(C->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(C->iIGates+n1stCell  +n)) * *(C->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = 0.0;
        *(C->eOGates+n1stCell+n) = err * sigm->evalDiff(*(C->iOGates+n1stCell  +n)) * func->eval(*(C->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = err * *(C->oOGates+n1stCell+n) * func->evalDiff(*(C->ostates +n1stCell +n));
    }
}

void LSTMLayer::backPropagateDelta(const Activation* const P, Activation* const C, const Activation* const N, const Real* const weights, const Real* const biases) const
{
    const Link* const lO = output_links;
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        
        //for (int k=0; k<output_links->size(); k++) {
        //    const Link* const lO = (*output_links)[k];
        err += propagateErrors(lO, C, n, weights);
        //}
        err += propagateErrors(recurrent_links, N, n, weights);

        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(C->in_vals+n1stNeuron+n)) * *(C->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(C->iIGates+n1stCell  +n)) * *(C->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = sigm->evalDiff(*(C->iFGates+n1stCell  +n)) * *(P->ostates+n1stCell+n);
        *(C->eOGates+n1stCell+n) = err * sigm->evalDiff(*(C->iOGates+n1stCell  +n)) * func->eval(*(C->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = err * *(C->oOGates+n1stCell+n) * func->evalDiff(*(C->ostates +n1stCell +n)) + *(N->errvals+n1stNeuron+n)* *(N->oFGates+n1stCell+n);
    }
}

void LSTMLayer::backPropagateDeltaLast(const Activation* const P, Activation* const C, const Real* const weights, const Real* const biases) const
{
    const Link* const lO = output_links;
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        
        //for (int k=0; k<output_links->size(); k++) {
        //    const Link* const lO = (*output_links)[k];
        err += propagateErrors(lO, C, n, weights);
        //}

        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(C->in_vals+n1stNeuron+n)) * *(C->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(C->iIGates+n1stCell  +n)) * *(C->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = sigm->evalDiff(*(C->iFGates+n1stCell  +n)) * *(P->ostates+n1stCell+n);
        *(C->eOGates+n1stCell+n) = err * sigm->evalDiff(*(C->iOGates+n1stCell  +n)) * func->eval(*(C->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = err * *(C->oOGates+n1stCell+n) * func->evalDiff(*(C->ostates +n1stCell +n));
    }
}

/* backPropagateGrads(Activation * prev, Activation * curr, Activation * next, Real* weights, Real* biases)
 *
 */

void NormalLayer::backPropagateGrads(const Activation* const P, const Activation* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) = eC;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            *(grad->_W +lI->iW +n*lI->nI +i) = *(C->outvals +lI->iI +i) * eC;
        }
        //}
        for (int i=0; i<lR->nI; i++) {
            *(grad->_W +lR->iW +n*lR->nI +i) = *(P->outvals +lR->iI +i) * eC;
        }
    }
}

void LSTMLayer::backPropagateGrads(const Activation* const P, const Activation* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) = eC; *(grad->_B +n1stBiasIG +n) = eI;
        *(grad->_B +n1stBiasFG +n) = eF; *(grad->_B +n1stBiasOG +n) = eO;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(C->outvals+lI->iI+i);
            *(grad->_W+lI->iW +n*lI->nI+i) = oVal * eC;
            *(grad->_W+lI->iWI+n*lI->nI+i) = oVal * eI;
            *(grad->_W+lI->iWF+n*lI->nI+i) = oVal * eF;
            *(grad->_W+lI->iWO+n*lI->nI+i) = oVal * eO;
        }
        //}
        for (int i=0; i<lR->nI; i++) {
            const Real oVal = *(P->outvals+lR->iI+i);
            *(grad->_W+lR->iW +n*lR->nI+i) = oVal * eC;
            *(grad->_W+lR->iWI+n*lR->nI+i) = oVal * eI;
            *(grad->_W+lR->iWF+n*lR->nI+i) = oVal * eF;
            *(grad->_W+lR->iWO+n*lR->nI+i) = oVal * eO;
        }
    }
}

void NormalLayer::backPropagateGrads(const Activation* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) = eC;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            *(grad->_W +lI->iW +n*lI->nI +i) = *(C->outvals +lI->iI +i) * eC;
        }
        //}
    }
}

void LSTMLayer::backPropagateGrads(const Activation* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) = eC; *(grad->_B +n1stBiasIG +n) = eI;
        *(grad->_B +n1stBiasFG +n) = eF; *(grad->_B +n1stBiasOG +n) = eO;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(C->outvals+lI->iI+i);
            *(grad->_W+lI->iW +n*lI->nI+i) = oVal * eC;
            *(grad->_W+lI->iWI+n*lI->nI+i) = oVal * eI;
            *(grad->_W+lI->iWF+n*lI->nI+i) = oVal * eF;
            *(grad->_W+lI->iWO+n*lI->nI+i) = oVal * eO;
        }
        //}
    }
}

void NormalLayer::backPropagateAddGrads(const Activation* const P, const Activation* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) += eC;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            *(grad->_W +lI->iW +n*lI->nI +i) += *(C->outvals +lI->iI +i) * eC;
        }
        //}
        for (int i=0; i<lR->nI; i++) {
            *(grad->_W +lR->iW +n*lR->nI +i) += *(P->outvals +lR->iI +i) * eC;
        }
    }
}

void LSTMLayer::backPropagateAddGrads(const Activation* const P, const Activation* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) += eC; *(grad->_B +n1stBiasIG +n) += eI;
        *(grad->_B +n1stBiasFG +n) += eF; *(grad->_B +n1stBiasOG +n) += eO;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(C->outvals+lI->iI+i);
            *(grad->_W+lI->iW +n*lI->nI+i) += oVal * eC;
            *(grad->_W+lI->iWI+n*lI->nI+i) += oVal * eI;
            *(grad->_W+lI->iWF+n*lI->nI+i) += oVal * eF;
            *(grad->_W+lI->iWO+n*lI->nI+i) += oVal * eO;
        }
        //}
        for (int i=0; i<lR->nI; i++) {
            const Real oVal = *(P->outvals+lR->iI+i);
            *(grad->_W+lR->iW +n*lR->nI+i) += oVal * eC;
            *(grad->_W+lR->iWI+n*lR->nI+i) += oVal * eI;
            *(grad->_W+lR->iWF+n*lR->nI+i) += oVal * eF;
            *(grad->_W+lR->iWO+n*lR->nI+i) += oVal * eO;
        }
    }
}

void NormalLayer::backPropagateAddGrads(const Activation* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) = eC;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            *(grad->_W +lI->iW +n*lI->nI +i) += *(C->outvals +lI->iI +i) * eC;
        }
        //}
    }
}

void LSTMLayer::backPropagateAddGrads(const Activation* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) = eC; *(grad->_B +n1stBiasIG +n) = eI;
        *(grad->_B +n1stBiasFG +n) = eF; *(grad->_B +n1stBiasOG +n) = eO;
        
        //for (int k=0; k<input_links->size(); k++) {
        //    const Link* const lI = (*input_links)[k];
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(C->outvals+lI->iI+i);
            *(grad->_W+lI->iW +n*lI->nI+i) += oVal * eC;
            *(grad->_W+lI->iWI+n*lI->nI+i) += oVal * eI;
            *(grad->_W+lI->iWF+n*lI->nI+i) += oVal * eF;
            *(grad->_W+lI->iWO+n*lI->nI+i) += oVal * eO;
        }
        //}
    }
}