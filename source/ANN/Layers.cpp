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

void NormalLayer::propagate(const Mem* const M, Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    Real in(0);
    #ifdef SIMDKERNELSIN
    Real * inry;
    _allocateQuick(inry, SIMD)
    #endif
    
    for (int n=0; n<nNeurons; n++) {
        in = 0.; //zero the first

        #ifdef SIMDKERNELSIN
        vec IN=SET0();
        for (int i=0; i<lI->nI; i+=SIMD) {
            IN = ADD(IN, MUL(LOAD(outvals +lI->iI +i), LOAD(weights +lI->iW  +n*lI->nI +i)));
        }
        for (int i=0; i<lR->nI; i+=SIMD) {
            IN = ADD(IN, MUL(LOAD(M->outvals +lR->iI +i), LOAD(weights +lR->iW  +n*lR->nI +i)));
        }
        STORE (inry,IN);
        for(int t=0; t<SIMD; t++) in += *(inry+t);
        #else
        for (int i=0; i<lI->nI; i++) {
            in += *(outvals +lI->iI +i) * *(weights +lI->iW +n*lI->nI +i);
        }
        for (int i=0; i<lR->nI; i++) {
            in += *(M->outvals +lR->iI +i) * *(weights +lR->iW +n*lR->nI +i);
        }
        #endif

        in += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = in; //first element of in
        *(N->outvals +n1stNeuron +n) = func->eval( in );
    }
    #ifdef SIMDKERNELSIN
    _myfree(inry);
    #endif
}

void NormalLayer::propagate(const Lab* const M, Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    Real in(0);
    #ifdef SIMDKERNELSIN
    Real * inry;
    _allocateQuick(inry, SIMD)
    #endif
    
    for (int n=0; n<nNeurons; n++) {
        in = 0.; //zero the first

        #ifdef SIMDKERNELSIN
        vec IN=SET0();
        for (int i=0; i<lI->nI; i+=SIMD) {
            IN = ADD(IN, MUL(LOAD(outvals +lI->iI +i), LOAD(weights +lI->iW  +n*lI->nI +i)));
        }
        for (int i=0; i<lR->nI; i+=SIMD) {
            IN = ADD(IN, MUL(LOAD(M->outvals +lR->iI +i), LOAD(weights +lR->iW  +n*lR->nI +i)));
        }
        STORE (inry,IN);
        for(int t=0; t<SIMD; t++) in += *(inry+t);
        #else
        for (int i=0; i<lI->nI; i++) {
            in += *(outvals +lI->iI +i) * *(weights +lI->iW +n*lI->nI +i);
        }
        for (int i=0; i<lR->nI; i++) {
            in += *(M->outvals +lR->iI +i) * *(weights +lR->iW +n*lR->nI +i);
        }
        #endif

        in += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = in; //first element of in
        *(N->outvals +n1stNeuron +n) = func->eval( in );
    }
    #ifdef SIMDKERNELSIN
    _myfree(inry);
    #endif
}

void NormalLayer::propagate(Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lI = input_links;
    Real in(0);
    #ifdef SIMDKERNELSIN
    Real * inry;
    _allocateQuick(inry, SIMD)
    #endif
    for (int n=0; n<nNeurons; n++) {
        in = 0.; //zero the first

        #ifdef SIMDKERNELSIN
        vec IN=SET0();
        for (int i=0; i<lI->nI; i+=SIMD) {
            IN = ADD(IN, MUL(LOAD(outvals +lI->iI +i), LOAD(weights +lI->iW  +n*lI->nI +i)));
        }
        STORE (inry,IN);
        for(int t=0; t<SIMD; t++) in += *(inry+t);
        #else
        for (int i=0; i<lI->nI; i++) {
            in += *(outvals +lI->iI +i) * *(weights +lI->iW +n*lI->nI +i);
        }
        #endif

        in += *(biases +n1stBias +n);
        *(N->in_vals +n1stNeuron +n) = in; //first element of in
        *(N->outvals +n1stNeuron +n) = func->eval( in );
    }
    #ifdef SIMDKERNELSIN
    _myfree(inry);
    #endif
}

void LSTMLayer::propagate(Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lI = input_links;
    Real tC(0), tI(0), tF(0), tO(0);
    #ifdef SIMDKERNELSIN
    Real *tCry,*tIry,*tFry,*tOry;
    _allocateQuick(tCry, SIMD)
    _allocateQuick(tIry, SIMD)
    _allocateQuick(tFry, SIMD)
    _allocateQuick(tOry, SIMD)
    #endif
    for (int n=0; n<nNeurons; n++) {
        tC=0.; tI=0.; tF=0.; tO=0.;
        #ifdef SIMDKERNELSIN
        vec IN=SET0(); vec IG=SET0(); vec FG=SET0(); vec OG=SET0();
        for (int i=0; i<lI->nI; i+=SIMD) {
            const vec O = LOAD(outvals +lI->iI +i);
            IN = ADD(IN, MUL(O, LOAD(weights +lI->iW  +n*lI->nI +i)));
            IG = ADD(IG, MUL(O, LOAD(weights +lI->iWI +n*lI->nI +i)));
            FG = ADD(FG, MUL(O, LOAD(weights +lI->iWF +n*lI->nI +i)));
            OG = ADD(OG, MUL(O, LOAD(weights +lI->iWO +n*lI->nI +i)));
        }
        STORE (tCry,IN); STORE (tIry,IG); STORE (tFry,FG); STORE (tOry,OG);
        for(int t=0; t<SIMD; t++) {
            tC += *(tCry+t); tI += *(tIry+t); tF += *(tFry+t); tO += *(tOry+t);
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(outvals +lI->iI +i);
            tC += oVal * *(weights +lI->iW  +n*lI->nI +i);
            tI += oVal * *(weights +lI->iWI +n*lI->nI +i);
            tF += oVal * *(weights +lI->iWF +n*lI->nI +i);
            tO += oVal * *(weights +lI->iWO +n*lI->nI +i);
        }
        #endif
        tC += *(biases +n1stBias +n);   *(N->in_vals +n1stNeuron +n) = tC;
        tI += *(biases +n1stBiasIG +n); *(N->iIGates +n1stCell   +n) = tI;//*(oldstates +n1stCell +n)* *(weights +n1stPeep +3*n)   + ;
        tF += *(biases +n1stBiasFG +n); *(N->iFGates +n1stCell   +n) = tF;//*(oldstates +n1stCell +n)* *(weights +n1stPeep +3*n +1)+
        const Real oC = ifun->eval(tC); *(N->oMCell  +n1stCell +n) = oC;
        const Real oI = sigm->eval(tI); *(N->oIGates +n1stCell +n) = oI;
        const Real oF = sigm->eval(tF); *(N->oFGates +n1stCell +n) = oF;
        const Real oS = oC * oI;        *(N->ostates +n1stCell +n) = oS;
        tO += *(biases +n1stBiasOG +n); *(N->iOGates +n1stCell +n) = tO;//*(N->ostates +n1stCell +n) * *(weights +n1stPeep +3*n +2) +
        const Real oO = sigm->eval(tO); *(N->oOGates +n1stCell +n) = oO;
        *(N->outvals +n1stNeuron +n) = func->eval(oS) * oO;
    }
    #ifdef SIMDKERNELSIN
    _myfree(tCry); _myfree(tIry); _myfree(tFry); _myfree(tOry);
    #endif
}

void LSTMLayer::propagate(const Mem* const M, Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    Real tC(0), tI(0), tF(0), tO(0);
    #ifdef SIMDKERNELSIN
    Real *tCry,*tIry,*tFry,*tOry;
    _allocateQuick(tCry, SIMD)
    _allocateQuick(tIry, SIMD)
    _allocateQuick(tFry, SIMD)
    _allocateQuick(tOry, SIMD)
    #endif
    for (int n=0; n<nNeurons; n++) {
        tC=0.; tI=0.; tF=0.; tO=0.;
        #ifdef SIMDKERNELSIN
        vec IN=SET0(); vec IG=SET0(); vec FG=SET0(); vec OG=SET0();
        for (int i=0; i<lI->nI; i+=SIMD) {
            const vec O = LOAD(outvals +lI->iI +i);
            IN = ADD(IN, MUL(O, LOAD(weights +lI->iW  +n*lI->nI +i)));
            IG = ADD(IG, MUL(O, LOAD(weights +lI->iWI +n*lI->nI +i)));
            FG = ADD(FG, MUL(O, LOAD(weights +lI->iWF +n*lI->nI +i)));
            OG = ADD(OG, MUL(O, LOAD(weights +lI->iWO +n*lI->nI +i)));
        }
        for (int i=0; i<lR->nI; i+=SIMD) {
            const vec O = LOAD(M->outvals +lR->iI +i);
            IN = ADD(IN, MUL(O, LOAD(weights +lR->iW  +n*lR->nI +i)));
            IG = ADD(IG, MUL(O, LOAD(weights +lR->iWI +n*lR->nI +i)));
            FG = ADD(FG, MUL(O, LOAD(weights +lR->iWF +n*lR->nI +i)));
            OG = ADD(OG, MUL(O, LOAD(weights +lR->iWO +n*lR->nI +i)));
        }
        STORE (tCry,IN); STORE (tIry,IG); STORE (tFry,FG); STORE (tOry,OG);
        for(int t=0; t<SIMD; t++) {
            tC += *(tCry+t); tI += *(tIry+t); tF += *(tFry+t); tO += *(tOry+t);
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(outvals +lI->iI +i);
            tC += oVal * *(weights +lI->iW  +n*lI->nI +i);
            tI += oVal * *(weights +lI->iWI +n*lI->nI +i);
            tF += oVal * *(weights +lI->iWF +n*lI->nI +i);
            tO += oVal * *(weights +lI->iWO +n*lI->nI +i);
        }
        for (int i=0; i<lR->nI; i++) {
            const Real oVal = *(M->outvals +lR->iI +i);
            tC += oVal * *(weights +lR->iW  +n*lR->nI +i);
            tI += oVal * *(weights +lR->iWI +n*lR->nI +i);
            tF += oVal * *(weights +lR->iWF +n*lR->nI +i);
            tO += oVal * *(weights +lR->iWO +n*lR->nI +i);
        }
        #endif
        
        tC += *(biases +n1stBias +n); *(N->in_vals +n1stNeuron +n) = tC;
        tI += *(biases +n1stBiasIG +n); *(N->iIGates +n1stCell   +n) = tI;//*(oldstates +n1stCell +n)* *(weights +n1stPeep +3*n)   + ;
        tF += *(biases +n1stBiasFG +n); *(N->iFGates +n1stCell   +n) = tF;//*(oldstates +n1stCell +n)* *(weights +n1stPeep +3*n +1)+
        const Real oC = ifun->eval(tC); *(N->oMCell  +n1stCell +n) = oC;
        const Real oI = sigm->eval(tI); *(N->oIGates +n1stCell +n) = oI;
        const Real oF = sigm->eval(tF); *(N->oFGates +n1stCell +n) = oF;
        const Real oS = *(M->ostates +n1stCell +n) * oF + oC * oI; *(N->ostates +n1stCell +n) = oS;
        tO += *(biases +n1stBiasOG +n); *(N->iOGates +n1stCell +n) = tO;//*(N->ostates +n1stCell +n) * *(weights +n1stPeep +3*n +2) +
        const Real oO = sigm->eval(tO); *(N->oOGates +n1stCell +n) = oO;
        *(N->outvals +n1stNeuron +n) = func->eval(oS) * oO;
    }
    #ifdef SIMDKERNELSIN
    _myfree(tCry); _myfree(tIry); _myfree(tFry); _myfree(tOry);
    #endif
}

void LSTMLayer::propagate(const Lab* const M, Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Real* const outvals = N->outvals;
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    Real tC(0), tI(0), tF(0), tO(0);
    #ifdef SIMDKERNELSIN
    Real *tCry,*tIry,*tFry,*tOry;
    _allocateQuick(tCry, SIMD)
    _allocateQuick(tIry, SIMD)
    _allocateQuick(tFry, SIMD)
    _allocateQuick(tOry, SIMD)
    #endif
    for (int n=0; n<nNeurons; n++) {
        tC=0.; tI=0.; tF=0.; tO=0.;
        #ifdef SIMDKERNELSIN
        vec IN=SET0(); vec IG=SET0(); vec FG=SET0(); vec OG=SET0();
        for (int i=0; i<lI->nI; i+=SIMD) {
            const vec O = LOAD(outvals +lI->iI +i);
            IN = ADD(IN, MUL(O, LOAD(weights +lI->iW  +n*lI->nI +i)));
            IG = ADD(IG, MUL(O, LOAD(weights +lI->iWI +n*lI->nI +i)));
            FG = ADD(FG, MUL(O, LOAD(weights +lI->iWF +n*lI->nI +i)));
            OG = ADD(OG, MUL(O, LOAD(weights +lI->iWO +n*lI->nI +i)));
        }
        for (int i=0; i<lR->nI; i+=SIMD) {
            const vec O = LOAD(M->outvals +lR->iI +i);
            IN = ADD(IN, MUL(O, LOAD(weights +lR->iW  +n*lR->nI +i)));
            IG = ADD(IG, MUL(O, LOAD(weights +lR->iWI +n*lR->nI +i)));
            FG = ADD(FG, MUL(O, LOAD(weights +lR->iWF +n*lR->nI +i)));
            OG = ADD(OG, MUL(O, LOAD(weights +lR->iWO +n*lR->nI +i)));
        }
        STORE (tCry,IN); STORE (tIry,IG); STORE (tFry,FG); STORE (tOry,OG);
        for(int t=0; t<SIMD; t++) {
            tC += *(tCry+t); tI += *(tIry+t); tF += *(tFry+t); tO += *(tOry+t);
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(outvals +lI->iI +i);
            tC += oVal * *(weights +lI->iW  +n*lI->nI +i);
            tI += oVal * *(weights +lI->iWI +n*lI->nI +i);
            tF += oVal * *(weights +lI->iWF +n*lI->nI +i);
            tO += oVal * *(weights +lI->iWO +n*lI->nI +i);
        }
        for (int i=0; i<lR->nI; i++) {
            const Real oVal = *(M->outvals +lR->iI +i);
            tC += oVal * *(weights +lR->iW  +n*lR->nI +i);
            tI += oVal * *(weights +lR->iWI +n*lR->nI +i);
            tF += oVal * *(weights +lR->iWF +n*lR->nI +i);
            tO += oVal * *(weights +lR->iWO +n*lR->nI +i);
        }
        #endif
        
        tC += *(biases +n1stBias +n);   *(N->in_vals +n1stNeuron +n) = tC;
        tI += *(biases +n1stBiasIG +n); *(N->iIGates +n1stCell   +n) = tI;
        tF += *(biases +n1stBiasFG +n); *(N->iFGates +n1stCell   +n) = tF;
        tO += *(biases +n1stBiasOG +n); *(N->iOGates +n1stCell   +n) = tO;
        const Real oC = ifun->eval(tC); *(N->oMCell  +n1stCell +n) = oC;
        const Real oI = sigm->eval(tI); *(N->oIGates +n1stCell +n) = oI;
        const Real oF = sigm->eval(tF); *(N->oFGates +n1stCell +n) = oF;
        const Real oS = *(M->ostates +n1stCell +n) * oF + oC * oI; *(N->ostates +n1stCell +n) = oS;
        const Real oO = sigm->eval(tO); *(N->oOGates +n1stCell +n) = oO;
        *(N->outvals +n1stNeuron +n) = func->eval(oS) * oO;
    }
    #ifdef SIMDKERNELSIN
    _myfree(tCry); _myfree(tIry); _myfree(tFry); _myfree(tOry);
    #endif
}

/* backPropagate(Mem * M, Lab * N, Dsdw * dsdw, Grads * grad, Real* weights, Real* biases)
 *
 */
/*
void NormalLayer::backPropagate(const Lab* const P, Lab* const C, Grads* const grad, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        
        for (const auto & l : *curr_output_links) {
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(C->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                    *(C->errvals +l->iO +i) * (
                       *(C->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                       *(C->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                       *(C->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err  += *(C->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
        }
        
        const Real eC = err * func->evalDiff(*(C->in_vals +n1stNeuron+n));
        *(C->errvals +n1stNeuron +n) = eC;
        *(grad->_B +n1stBias +n) = eC;
        
        for (const auto & l : *curr_input_links) {
            #ifdef SIMDKERNELSG
            const vec E = BCAST(&eC);
            for (int i=0; i<l->nI; i+=SIMD) {
                STORE(grad->_W +l->iW +n*l->nI +i, MUL(E, LOAD(C->outvals +l->iI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                *(grad->_W +l->iW +n*l->nI +i) = *(C->outvals +l->iI +i) * eC;
            }
            #endif
        }
        for (const auto & l : *prev_input_links) {
            #ifdef SIMDKERNELSG
            const vec E = BCAST(&eC);
            for (int i=0; i<l->nI; i+=SIMD) {
                STORE(grad->_W +l->iW +n*l->nI +i, MUL(E, LOAD(P->outvals +l->iI +i)));
            }
            #else
            
            for (int i=0; i<l->nI; i++) {
                *(grad->_W +l->iW +n*l->nI +i) = *(P->outvals +l->iI +i) * eC;
            }
            #endif
        }
    }
}

void LSTMLayer::backPropagate(const Lab* const P, Lab* const C, Grads* const grad, const Real* const weights, const Real* const biases) const
{
    for (int n=0; n<nNeurons; n++)
    {
        Real err = (last) ? *(C->errvals +n1stNeuron +n) : 0.0;
        
        for (const auto & l : *curr_output_links)
        {
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(C->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                    *(C->errvals +l->iO +i) * (
                       *(C->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                       *(C->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                       *(C->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err  += *(C->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
        }
        
        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(C->in_vals+n1stNeuron+n)) * *(C->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(C->iIGates+n1stCell  +n)) * *(C->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = sigm->evalDiff(*(C->iFGates+n1stCell  +n)) * *(P->ostates+n1stCell+n);
        *(C->eOGates+n1stCell+n) = err * sigm->evalDiff(*(C->iOGates+n1stCell  +n)) * func->eval(*(C->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = err * *(C->oOGates+n1stCell+n) * func->evalDiff(*(C->ostates +n1stCell +n));
        
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates+n1stCell+n);
        *(grad->_B +n1stBias   +n) = eC; *(grad->_B +n1stBiasIG +n) = eI;
        *(grad->_B +n1stBiasFG +n) = eF; *(grad->_B +n1stBiasOG +n) = eO;
        
        for (const auto & l : *curr_input_links) {
            #ifdef SIMDKERNELSG
            const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
            const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
            for (int i=0; i<l->nI; i+=SIMD)
            {
                const vec O  = LOAD(C->outvals +l->iI +i);
                STORE(grad->_W +l->iW  +n*l->nI +i, MUL(EN,O));
                STORE(grad->_W +l->iWI +n*l->nI +i, MUL(EI,O));
                STORE(grad->_W +l->iWF +n*l->nI +i, MUL(EF,O));
                STORE(grad->_W +l->iWO +n*l->nI +i, MUL(EO,O));
            }
            #else
            for (int i=0; i<l->nI; i++)
            {
                const Real oVal = *(C->outvals+l->iI+i);
                *(grad->_W+l->iW +n*l->nI+i) = oVal * eC;
                *(grad->_W+l->iWI+n*l->nI+i) = oVal * eI;
                *(grad->_W+l->iWF+n*l->nI+i) = oVal * eF;
                *(grad->_W+l->iWO+n*l->nI+i) = oVal * eO;
            }
            #endif
        }
        for (const auto & l : *prev_input_links) {
            #ifdef SIMDKERNELSG
            const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
            const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O  = LOAD(P->outvals +l->iI +i);
                STORE(grad->_W +l->iW  +n*l->nI +i, MUL(EN,O));
                STORE(grad->_W +l->iWI +n*l->nI +i, MUL(EI,O));
                STORE(grad->_W +l->iWF +n*l->nI +i, MUL(EF,O));
                STORE(grad->_W +l->iWO +n*l->nI +i, MUL(EO,O));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(P->outvals+l->iI+i);
                *(grad->_W+l->iW +n*l->nI+i) = oVal * eC;
                *(grad->_W+l->iWI+n*l->nI+i) = oVal * eI;
                *(grad->_W+l->iWF+n*l->nI+i) = oVal * eF;
                *(grad->_W+l->iWO+n*l->nI+i) = oVal * eO;
            }
            #endif
        }
    }
}
*/
/* backPropagateDelta(Lab * prev, Lab * curr, Lab * next, Real* weights, Real* biases)
 *
 */

void NormalLayer::backPropagateDelta(Lab* const C, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    const Link* const lO = output_links;
    
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        if (lO->LSTM)
            for (int i=0; i<lO->nO; i++)
                err += *(cC->eOGates +lO->iC +i) * *(weights +lO->iWO +i*lO->nI +n) +
                       *(cC->errvals +lO->iO +i) * (
                                    *(cC->eMCell  +lO->iC +i) * *(weights +lO->iW  +i*lO->nI +n) +
                                    *(cC->eIGates +lO->iC +i) * *(weights +lO->iWI +i*lO->nI +n) +
                                    *(cC->eFGates +lO->iC +i) * *(weights +lO->iWF +i*lO->nI +n) );
        else
            for (int i=0; i<lO->nO; i++)
                err  += *(cC->errvals +lO->iO +i) * *(weights +lO->iW +i*lO->nI +n);

        *(C->errvals +n1stNeuron +n) = err * func->evalDiff(*(cC->in_vals +n1stNeuron+n));
    }
}

void NormalLayer::backPropagateDeltaFirst(Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    const Link* const lO = output_links;
    const Link* const lR = recurrent_links;
    
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        if (lO->LSTM)
            for (int i=0; i<lO->nO; i++)
                err += *(cC->eOGates +lO->iC +i) * *(weights +lO->iWO +i*lO->nI +n) +
                       *(cC->errvals +lO->iO +i) * (
                                     *(cC->eMCell  +lO->iC +i) * *(weights +lO->iW  +i*lO->nI +n) +
                                     *(cC->eIGates +lO->iC +i) * *(weights +lO->iWI +i*lO->nI +n) +
                                     *(cC->eFGates +lO->iC +i) * *(weights +lO->iWF +i*lO->nI +n) );
        else
            for (int i=0; i<lO->nO; i++)
                err  += *(cC->errvals +lO->iO +i) * *(weights +lO->iW +i*lO->nI +n);
        
        
        for (int i=0; i<lR->nO; i++)
            err  += *(N->errvals +lR->iO +i) * *(weights +lR->iW +i*lR->nI +n);
        
        *(C->errvals +n1stNeuron +n) = err * func->evalDiff(*(cC->in_vals +n1stNeuron+n));
    }
}

void LSTMLayer::backPropagateDeltaFirst(Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    const Link* const lO = output_links;
    const Link* const lR = recurrent_links;
    printf("Last delta with 1stN %d, 1stC %d, 1stB %d",n1stNeuron,n1stCell,n1stBias);
    lO->print();
    lR->print();
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        
        if (lO->LSTM) {
            for (int i=0; i<lO->nO; i++)
                err += *(cC->eOGates +lO->iC +i) * *(weights +lO->iWO +i*lO->nI +n) +
                *(cC->errvals +lO->iO +i) * (
                             *(cC->eMCell  +lO->iC +i) * *(weights +lO->iW  +i*lO->nI +n) +
                             *(cC->eIGates +lO->iC +i) * *(weights +lO->iWI +i*lO->nI +n) +
                             *(cC->eFGates +lO->iC +i) * *(weights +lO->iWF +i*lO->nI +n) );
        } else {
            for (int i=0; i<lO->nO; i++)
                err  += *(cC->errvals +lO->iO +i) * *(weights +lO->iW +i*lO->nI +n);
        }

        for (int i=0; i<lO->nO; i++)
            err += *(cC->eOGates +lO->iC +i) * *(weights +lO->iWO +i*lO->nI +n) +
            *(cC->errvals +lO->iO +i) * (
                         *(cC->eMCell  +lO->iC +i) * *(weights +lO->iW  +i*lO->nI +n) +
                         *(cC->eIGates +lO->iC +i) * *(weights +lO->iWI +i*lO->nI +n) +
                         *(cC->eFGates +lO->iC +i) * *(weights +lO->iWF +i*lO->nI +n) );
        /* Alternative for less reads:
        const Real tC = *(cC->in_vals +n1stNeuron +n);
        const Real tI = *(cC->iIGates +n1stCell   +n);
        const Real tF = *(cC->iFGates +n1stCell   +n);
        const Real tO = *(cC->iOGates +n1stCell   +n);
        const Real oS = *(cC->ostates+n1stCell+n);
        const Real oC = ifun->eval(tC);
        const Real oI = sigm->eval(tI);
        const Real oO = sigm->eval(tO);
        *(C->eMCell +n1stCell+n) = ifun->evalDiff(tC) * oI;
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(tI) * oC;
        *(C->eFGates+n1stCell+n) = 0.0;
        *(C->eOGates+n1stCell+n) = sigm->evalDiff(tO) * func->eval(oS) * err;
        *(C->errvals+n1stNeuron+n) = err * oO * func->evalDiff(oS) + *(N->errvals+n1stNeuron+n)* *(N->oFGates+n1stCell+n);
        */
        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(cC->in_vals+n1stNeuron+n)) * *(cC->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(cC->iIGates+n1stCell  +n)) * *(cC->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = 0.0;
        *(C->eOGates+n1stCell+n) = err * sigm->evalDiff(*(cC->iOGates+n1stCell  +n)) * func->eval(*(cC->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = err * *(cC->oOGates+n1stCell+n) * func->evalDiff(*(cC->ostates +n1stCell +n)) + *(N->errvals+n1stNeuron+n)* *(N->oFGates+n1stCell+n);
    }
}

void LSTMLayer::backPropagateDelta(Lab* const C, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    const Link* const lO = output_links;
    for (int n=0; n<nNeurons; n++)
    {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        
        if (lO->LSTM) {
            for (int i=0; i<lO->nO; i++)
                err += *(cC->eOGates +lO->iC +i) * *(weights +lO->iWO +i*lO->nI +n) +
                *(cC->errvals +lO->iO +i) * (
                                             *(cC->eMCell  +lO->iC +i) * *(weights +lO->iW  +i*lO->nI +n) +
                                             *(cC->eIGates +lO->iC +i) * *(weights +lO->iWI +i*lO->nI +n) +
                                             *(cC->eFGates +lO->iC +i) * *(weights +lO->iWF +i*lO->nI +n) );
        } else {
            for (int i=0; i<lO->nO; i++)
                err  += *(cC->errvals +lO->iO +i) * *(weights +lO->iW +i*lO->nI +n);
        }
        
        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(cC->in_vals+n1stNeuron+n)) * *(cC->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(cC->iIGates+n1stCell  +n)) * *(cC->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = 0.0;
        *(C->eOGates+n1stCell+n) = err * sigm->evalDiff(*(cC->iOGates+n1stCell  +n)) * func->eval(*(cC->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = err * *(cC->oOGates+n1stCell+n) * func->evalDiff(*(cC->ostates +n1stCell +n));
    }
}

void LSTMLayer::backPropagateDelta(const Lab* const P, Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    const Link* const lO = output_links;
    const Link* const lR = recurrent_links;
    printf("Normal delta with 1stN %d, 1stC %d, 1stB %d",n1stNeuron,n1stCell,n1stBias);
    lO->print();
    lR->print();
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        
        if (lO->LSTM) {
            for (int i=0; i<lO->nO; i++)
                err += *(cC->eOGates +lO->iC +i) * *(weights +lO->iWO +i*lO->nI +n) +
                *(cC->errvals +lO->iO +i) * (
                                             *(cC->eMCell  +lO->iC +i) * *(weights +lO->iW  +i*lO->nI +n) +
                                             *(cC->eIGates +lO->iC +i) * *(weights +lO->iWI +i*lO->nI +n) +
                                             *(cC->eFGates +lO->iC +i) * *(weights +lO->iWF +i*lO->nI +n) );
        } else {
            for (int i=0; i<lO->nO; i++)
                err  += *(cC->errvals +lO->iO +i) * *(weights +lO->iW +i*lO->nI +n);
        }
        for (int i=0; i<lO->nO; i++)
            err += *(cC->eOGates +lO->iC +i) * *(weights +lO->iWO +i*lO->nI +n) +
            *(cC->errvals +lO->iO +i) * (
                                         *(cC->eMCell  +lO->iC +i) * *(weights +lO->iW  +i*lO->nI +n) +
                                         *(cC->eIGates +lO->iC +i) * *(weights +lO->iWI +i*lO->nI +n) +
                                         *(cC->eFGates +lO->iC +i) * *(weights +lO->iWF +i*lO->nI +n) );
        /* Alternative for less reads:
        const Real tC = *(cC->in_vals +n1stNeuron +n);
        const Real tI = *(cC->iIGates +n1stCell   +n);
        const Real tF = *(cC->iFGates +n1stCell   +n);
        const Real tO = *(cC->iOGates +n1stCell   +n);
        const Real oS = *(cC->ostates+n1stCell+n);
        const Real oC = ifun->eval(tC);
        const Real oI = sigm->eval(tI);
        const Real oO = sigm->eval(tO);
        *(C->eMCell +n1stCell+n) = ifun->evalDiff(tC) * oI;
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(tI) * oC;
        *(C->eFGates+n1stCell+n) = sigm->evalDiff(tI) * *(P->ostates+n1stCell+n);
        *(C->eOGates+n1stCell+n) = sigm->evalDiff(tO) * func->eval(oS) * err;
        *(C->errvals+n1stNeuron+n) = err * oO * func->evalDiff(oS) + *(N->errvals+n1stNeuron+n)* *(N->oFGates+n1stCell+n);
        */
        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(cC->in_vals+n1stNeuron+n)) * *(cC->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(cC->iIGates+n1stCell  +n)) * *(cC->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = sigm->evalDiff(*(cC->iFGates+n1stCell  +n)) * *(P->ostates+n1stCell+n);
        *(C->eOGates+n1stCell+n) = err * sigm->evalDiff(*(cC->iOGates+n1stCell  +n)) * func->eval(*(cC->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = err * *(cC->oOGates+n1stCell+n) * func->evalDiff(*(cC->ostates +n1stCell +n)) + *(N->errvals+n1stNeuron+n)* *(N->oFGates+n1stCell+n);// +
        //*(next->eIGates+n1stCell+n)* *(weights+n1stPeep+3*n)   +
        //*(next->eFGates+n1stCell+n)* *(weights+n1stPeep+3*n+1) +
         //*(curr->eOGates+n1stCell+n)* *(weights+n1stPeep+3*n+2);
    }
}

void LSTMLayer::backPropagateDeltaLast(const Lab* const P, Lab* const C, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    const Link* const lO = output_links;
    printf("Last delta with 1stN %d, 1stC %d, 1stB %d",n1stNeuron,n1stCell,n1stBias);
    lO->print();
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        
        if (lO->LSTM) {
            for (int i=0; i<lO->nO; i++)
                err += *(cC->eOGates +lO->iC +i) * *(weights +lO->iWO +i*lO->nI +n) +
                *(cC->errvals +lO->iO +i) * (
                                             *(cC->eMCell  +lO->iC +i) * *(weights +lO->iW  +i*lO->nI +n) +
                                             *(cC->eIGates +lO->iC +i) * *(weights +lO->iWI +i*lO->nI +n) +
                                             *(cC->eFGates +lO->iC +i) * *(weights +lO->iWF +i*lO->nI +n) );
        } else {
            for (int i=0; i<lO->nO; i++)
                err  += *(cC->errvals +lO->iO +i) * *(weights +lO->iW +i*lO->nI +n);
        }
        
        *(C->eMCell +n1stCell+n) = ifun->evalDiff(*(cC->in_vals+n1stNeuron+n)) * *(cC->oIGates+n1stCell+n);
        *(C->eIGates+n1stCell+n) = sigm->evalDiff(*(cC->iIGates+n1stCell  +n)) * *(cC->oMCell +n1stCell+n);
        *(C->eFGates+n1stCell+n) = sigm->evalDiff(*(cC->iFGates+n1stCell  +n)) * *(P->ostates+n1stCell+n);
        *(C->eOGates+n1stCell+n) = err * sigm->evalDiff(*(cC->iOGates+n1stCell  +n)) * func->eval(*(cC->ostates+n1stCell+n));
        *(C->errvals+n1stNeuron+n) = err * *(cC->oOGates+n1stCell+n) * func->evalDiff(*(cC->ostates +n1stCell +n));
    }
}

/* backPropagateGrads(Lab * prev, Lab * curr, Lab * next, Real* weights, Real* biases)
 *
 */

void NormalLayer::backPropagateGrads(const Lab* const P, const Lab* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) = eC;
        #ifdef SIMDKERNELSG
        const vec E = BCAST(&eC);
        for (int i=0; i<lI->nI; i+=SIMD) {
            STORE(grad->_W +lI->iW +n*lI->nI +i, MUL(E, LOAD(C->outvals +lI->iI +i)));
        }
        for (int i=0; i<lR->nI; i+=SIMD) {
            STORE(grad->_W +lR->iW +n*lR->nI +i, MUL(E, LOAD(P->outvals +lR->iI +i)));
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            *(grad->_W +lI->iW +n*lI->nI +i) = *(C->outvals +lI->iI +i) * eC;
        }
        for (int i=0; i<lR->nI; i++) {
            *(grad->_W +lR->iW +n*lR->nI +i) = *(P->outvals +lR->iI +i) * eC;
        }
        #endif
    }
}

void LSTMLayer::backPropagateGrads(const Lab* const P, const Lab* const C, Grads* const grad) const
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
        
        #ifdef SIMDKERNELSG
        const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
        const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
        for (int i=0; i<lI->nI; i+=SIMD) {
            const vec O  = LOAD(C->outvals +lI->iI +i);
            STORE(grad->_W +lI->iW  +n*lI->nI +i, MUL(EN,O));
            STORE(grad->_W +lI->iWI +n*lI->nI +i, MUL(EI,O));
            STORE(grad->_W +lI->iWF +n*lI->nI +i, MUL(EF,O));
            STORE(grad->_W +lI->iWO +n*lI->nI +i, MUL(EO,O));
        }
        for (int i=0; i<lR->nI; i+=SIMD) {
            const vec O  = LOAD(P->outvals +lR->iI +i);
            STORE(grad->_W +lR->iW  +n*lR->nI +i, MUL(EN,O));
            STORE(grad->_W +lR->iWI +n*lR->nI +i, MUL(EI,O));
            STORE(grad->_W +lR->iWF +n*lR->nI +i, MUL(EF,O));
            STORE(grad->_W +lR->iWO +n*lR->nI +i, MUL(EO,O));
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(C->outvals+lI->iI+i);
            *(grad->_W+lI->iW +n*lI->nI+i) = oVal * eC;
            *(grad->_W+lI->iWI+n*lI->nI+i) = oVal * eI;
            *(grad->_W+lI->iWF+n*lI->nI+i) = oVal * eF;
            *(grad->_W+lI->iWO+n*lI->nI+i) = oVal * eO;
        }
        for (int i=0; i<lR->nI; i++) {
            const Real oVal = *(P->outvals+lR->iI+i);
            *(grad->_W+lR->iW +n*lR->nI+i) = oVal * eC;
            *(grad->_W+lR->iWI+n*lR->nI+i) = oVal * eI;
            *(grad->_W+lR->iWF+n*lR->nI+i) = oVal * eF;
            *(grad->_W+lR->iWO+n*lR->nI+i) = oVal * eO;
        }
        #endif
    }
}

void NormalLayer::backPropagateGrads(const Lab* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) = eC;
        
        #ifdef SIMDKERNELSG
        const vec E = BCAST(&eC);
        for (int i=0; i<lI->nI; i+=SIMD) {
            STORE(grad->_W +lI->iW +n*lI->nI +i, MUL(E, LOAD(C->outvals +lI->iI +i)));
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            *(grad->_W +lI->iW +n*lI->nI +i) = *(C->outvals +lI->iI +i) * eC;
        }
        #endif
    }
}

void LSTMLayer::backPropagateGrads(const Lab* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) = eC; *(grad->_B +n1stBiasIG +n) = eI;
        *(grad->_B +n1stBiasFG +n) = eF; *(grad->_B +n1stBiasOG +n) = eO;
        
        #ifdef SIMDKERNELSG
        const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
        const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
        for (int i=0; i<lI->nI; i+=SIMD) {
            const vec O  = LOAD(C->outvals +lI->iI +i);
            STORE(grad->_W +lI->iW  +n*lI->nI +i, MUL(EN,O));
            STORE(grad->_W +lI->iWI +n*lI->nI +i, MUL(EI,O));
            STORE(grad->_W +lI->iWF +n*lI->nI +i, MUL(EF,O));
            STORE(grad->_W +lI->iWO +n*lI->nI +i, MUL(EO,O));
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(C->outvals+lI->iI+i);
            *(grad->_W+lI->iW +n*lI->nI+i) = oVal * eC;
            *(grad->_W+lI->iWI+n*lI->nI+i) = oVal * eI;
            *(grad->_W+lI->iWF+n*lI->nI+i) = oVal * eF;
            *(grad->_W+lI->iWO+n*lI->nI+i) = oVal * eO;
        }
        #endif
    }
}

void NormalLayer::backPropagateAddGrads(const Lab* const P, const Lab* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    const Link* const lR = recurrent_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) += eC;
        
        #ifdef SIMDKERNELSG
        const vec E = BCAST(&eC);
        for (int i=0; i<lI->nI; i+=SIMD) {
            STORE(grad->_W +lI->iW +n*lI->nI +i,
                  ADD(LOAD(grad->_W +lI->iW +n*lI->nI +i),
                      MUL(E, LOAD(C->outvals +lI->iI +i))));
        }
        for (int i=0; i<lR->nI; i+=SIMD) {
            STORE(grad->_W +lR->iW +n*lR->nI +i,
                  ADD(LOAD(grad->_W +lR->iW +n*lR->nI +i),
                      MUL(E, LOAD(P->outvals +lR->iI +i))));
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            *(grad->_W +lI->iW +n*lI->nI +i) += *(C->outvals +lI->iI +i) * eC;
        }
        for (int i=0; i<lR->nI; i++) {
            *(grad->_W +lR->iW +n*lR->nI +i) += *(P->outvals +lR->iI +i) * eC;
        }
        #endif
    }
}

void LSTMLayer::backPropagateAddGrads(const Lab* const P, const Lab* const C, Grads* const grad) const
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
        
        #ifdef SIMDKERNELSG
        const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
        const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
        for (int i=0; i<lI->nI; i+=SIMD) {
            const vec O  = LOAD(C->outvals +lI->iI +i);
            
               STORE(grad->_W +lI->iW  +n*lI->nI +i,
            ADD(LOAD(grad->_W +lI->iW  +n*lI->nI +i), MUL(EN,O)));
               STORE(grad->_W +lI->iWI +n*lI->nI +i,
            ADD(LOAD(grad->_W +lI->iWI +n*lI->nI +i), MUL(EI,O)));
               STORE(grad->_W +lI->iWF +n*lI->nI +i,
            ADD(LOAD(grad->_W +lI->iWF +n*lI->nI +i), MUL(EF,O)));
               STORE(grad->_W +lI->iWO +n*lI->nI +i,
            ADD(LOAD(grad->_W +lI->iWO +n*lI->nI +i), MUL(EO,O)));
        }
        for (int i=0; i<lR->nI; i+=SIMD) {
            const vec O  = LOAD(P->outvals +lR->iI +i);
             STORE(grad->_W +lR->iW  +n*lR->nI +i,
          ADD(LOAD(grad->_W +lR->iW  +n*lR->nI +i), MUL(EN,O)));
             STORE(grad->_W +lR->iWI +n*lR->nI +i,
          ADD(LOAD(grad->_W +lR->iWI +n*lR->nI +i), MUL(EI,O)));
             STORE(grad->_W +lR->iWF +n*lR->nI +i,
          ADD(LOAD(grad->_W +lR->iWF +n*lR->nI +i), MUL(EF,O)));
             STORE(grad->_W +lR->iWO +n*lR->nI +i,
          ADD(LOAD(grad->_W +lR->iWO +n*lR->nI +i), MUL(EO,O)));
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(C->outvals+lI->iI+i);
            *(grad->_W+lI->iW +n*lI->nI+i) += oVal * eC;
            *(grad->_W+lI->iWI+n*lI->nI+i) += oVal * eI;
            *(grad->_W+lI->iWF+n*lI->nI+i) += oVal * eF;
            *(grad->_W+lI->iWO+n*lI->nI+i) += oVal * eO;
        }
        for (int i=0; i<lR->nI; i++) {
            const Real oVal = *(P->outvals+lR->iI+i);
            *(grad->_W+lR->iW +n*lR->nI+i) += oVal * eC;
            *(grad->_W+lR->iWI+n*lR->nI+i) += oVal * eI;
            *(grad->_W+lR->iWF+n*lR->nI+i) += oVal * eF;
            *(grad->_W+lR->iWO+n*lR->nI+i) += oVal * eO;
        }
        #endif
    }
}

void NormalLayer::backPropagateAddGrads(const Lab* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) = eC;
        #ifdef SIMDKERNELSG
        const vec E = BCAST(&eC);
        for (int i=0; i<lI->nI; i+=SIMD) {
            STORE(grad->_W +lI->iW +n*lI->nI +i,
                  ADD(LOAD(grad->_W +lI->iW +n*lI->nI +i),
                      MUL(E, LOAD(C->outvals +lI->iI +i))));
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            *(grad->_W +lI->iW +n*lI->nI +i) += *(C->outvals +lI->iI +i) * eC;
        }
        #endif
    }
}

void LSTMLayer::backPropagateAddGrads(const Lab* const C, Grads* const grad) const
{
    const Link* const lI = input_links;
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) = eC; *(grad->_B +n1stBiasIG +n) = eI;
        *(grad->_B +n1stBiasFG +n) = eF; *(grad->_B +n1stBiasOG +n) = eO;
        
        #ifdef SIMDKERNELSG
        const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
        const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
        for (int i=0; i<lI->nI; i+=SIMD) {
            const vec O  = LOAD(C->outvals +lI->iI +i);
            
               STORE(grad->_W +lI->iW  +n*lI->nI +i,
            ADD(LOAD(grad->_W +lI->iW  +n*lI->nI +i), MUL(EN,O)));
               STORE(grad->_W +lI->iWI +n*lI->nI +i,
            ADD(LOAD(grad->_W +lI->iWI +n*lI->nI +i), MUL(EI,O)));
               STORE(grad->_W +lI->iWF +n*lI->nI +i,
            ADD(LOAD(grad->_W +lI->iWF +n*lI->nI +i), MUL(EF,O)));
               STORE(grad->_W +lI->iWO +n*lI->nI +i,
            ADD(LOAD(grad->_W +lI->iWO +n*lI->nI +i), MUL(EO,O)));
        }
        #else
        for (int i=0; i<lI->nI; i++) {
            const Real oVal = *(C->outvals+lI->iI+i);
            *(grad->_W+lI->iW +n*lI->nI+i) += oVal * eC;
            *(grad->_W+lI->iWI+n*lI->nI+i) += oVal * eI;
            *(grad->_W+lI->iWF+n*lI->nI+i) += oVal * eF;
            *(grad->_W+lI->iWO+n*lI->nI+i) += oVal * eO;
        }
        #endif
    }
}