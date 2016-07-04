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
    Real in(0);
    #ifdef SIMDKERNELSIN
    Real * inry;
    _allocateQuick(inry, SIMD)
    #endif
    
    for (int n=0; n<nNeurons; n++) {
        in = 0.; //zero the first
        #ifdef SIMDKERNELSIN
        vec IN=SET0();
        #endif
        
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                IN = ADD(IN, MUL(LOAD(outvals +l->iI +i), LOAD(weights +l->iW  +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                in += *(outvals +l->iI +i) * *(weights +l->iW +n*l->nI +i);
            }
            #endif
        }
        {
            const Link* const l = recurrent_links;
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                IN = ADD(IN, MUL(LOAD(M->outvals +l->iI +i), LOAD(weights +l->iW  +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                in += *(M->outvals +l->iI +i) * *(weights +l->iW +n*l->nI +i);
            }
            #endif
        }
        
        #ifdef SIMDKERNELSIN
        STORE (inry,IN);
        for(int t=0; t<SIMD; t++) in += *(inry+t);
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
    Real in(0);
    #ifdef SIMDKERNELSIN
    Real * inry;
    _allocateQuick(inry, SIMD)
    #endif
    
    for (int n=0; n<nNeurons; n++) {
        in = 0.; //zero the first
        #ifdef SIMDKERNELSIN
        vec IN=SET0();
        #endif
        
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                IN = ADD(IN, MUL(LOAD(outvals +l->iI +i), LOAD(weights +l->iW  +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                in += *(outvals +l->iI +i) * *(weights +l->iW +n*l->nI +i);
            }
            #endif
        }
        {
            const Link* const l = recurrent_links;
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                IN = ADD(IN, MUL(LOAD(M->outvals +l->iI +i), LOAD(weights +l->iW  +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                in += *(M->outvals +l->iI +i) * *(weights +l->iW +n*l->nI +i);
            }
            #endif
        }
        
        #ifdef SIMDKERNELSIN
        STORE (inry,IN);
        for(int t=0; t<SIMD; t++) in += *(inry+t);
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
    Real in(0);
    #ifdef SIMDKERNELSIN
    Real * inry;
    _allocateQuick(inry, SIMD)
    #endif
    for (int n=0; n<nNeurons; n++) {
        in = 0.; //zero the first
        #ifdef SIMDKERNELSIN
        vec IN=SET0();
        #endif
        
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                IN = ADD(IN, MUL(LOAD(outvals +l->iI +i), LOAD(weights +l->iW  +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                in += *(outvals +l->iI +i) * *(weights +l->iW +n*l->nI +i);
            }
            #endif
        }
        
        #ifdef SIMDKERNELSIN
        STORE (inry,IN);
        for(int t=0; t<SIMD; t++) in += *(inry+t);
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
        #endif
        
        {
            const Link* const l = input_links;
            //printf("First prop curr_input_links with 1stN %d, 1stC %d, 1stB %d ",n1stNeuron,n1stCell,n1stBias); fflush(0);
            //l->print(); fflush(0);
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                //FETCH((char*) N->outvals +l->iI +i + M_PF_O, M_POL_O);
                //FETCH((char*) weights +l->iW  +n*l->nI +i + M_PF_W, M_POL_W);
                //FETCH((char*) weights +l->iWI +n*l->nI +i + M_PF_W, M_POL_W);
                //FETCH((char*) weights +l->iWF +n*l->nI +i + M_PF_W, M_POL_W);
                //FETCH((char*) weights +l->iWO +n*l->nI +i + M_PF_W, M_POL_W);
                const vec O = LOAD(outvals +l->iI +i);
                IN = ADD(IN, MUL(O, LOAD(weights +l->iW  +n*l->nI +i)));
                IG = ADD(IG, MUL(O, LOAD(weights +l->iWI +n*l->nI +i)));
                FG = ADD(FG, MUL(O, LOAD(weights +l->iWF +n*l->nI +i)));
                OG = ADD(OG, MUL(O, LOAD(weights +l->iWO +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(outvals +l->iI +i);
                tC += oVal * *(weights +l->iW  +n*l->nI +i);
                tI += oVal * *(weights +l->iWI +n*l->nI +i);
                tF += oVal * *(weights +l->iWF +n*l->nI +i);
                tO += oVal * *(weights +l->iWO +n*l->nI +i);
            }
            #endif
        }
        
        #ifdef SIMDKERNELSIN
        STORE (tCry,IN); STORE (tIry,IG); STORE (tFry,FG); STORE (tOry,OG);
        #pragma unroll
        for(int t=0; t<SIMD; t++) {
            tC += *(tCry+t); tI += *(tIry+t); tF += *(tFry+t); tO += *(tOry+t);
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
        #endif
        
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O = LOAD(outvals +l->iI +i);
                IN = ADD(IN, MUL(O, LOAD(weights +l->iW  +n*l->nI +i)));
                IG = ADD(IG, MUL(O, LOAD(weights +l->iWI +n*l->nI +i)));
                FG = ADD(FG, MUL(O, LOAD(weights +l->iWF +n*l->nI +i)));
                OG = ADD(OG, MUL(O, LOAD(weights +l->iWO +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(outvals +l->iI +i);
                tC += oVal * *(weights +l->iW  +n*l->nI +i);
                tI += oVal * *(weights +l->iWI +n*l->nI +i);
                tF += oVal * *(weights +l->iWF +n*l->nI +i);
                tO += oVal * *(weights +l->iWO +n*l->nI +i);
            }
            #endif
        }
        {
            const Link* const l = recurrent_links;
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O = LOAD(M->outvals +l->iI +i);
                IN = ADD(IN, MUL(O, LOAD(weights +l->iW  +n*l->nI +i)));
                IG = ADD(IG, MUL(O, LOAD(weights +l->iWI +n*l->nI +i)));
                FG = ADD(FG, MUL(O, LOAD(weights +l->iWF +n*l->nI +i)));
                OG = ADD(OG, MUL(O, LOAD(weights +l->iWO +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(M->outvals +l->iI +i);
                tC += oVal * *(weights +l->iW  +n*l->nI +i);
                tI += oVal * *(weights +l->iWI +n*l->nI +i);
                tF += oVal * *(weights +l->iWF +n*l->nI +i);
                tO += oVal * *(weights +l->iWO +n*l->nI +i);
            }
            #endif
        }
        
        #ifdef SIMDKERNELSIN
        STORE (tCry,IN); STORE (tIry,IG); STORE (tFry,FG); STORE (tOry,OG);
        #pragma unroll
        for(int t=0; t<SIMD; t++) {
            tC += *(tCry+t); tI += *(tIry+t); tF += *(tFry+t); tO += *(tOry+t);
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
        #endif
        
        {
            const Link* const l = input_links;
            printf("Normal curr_input_links with 1stN %d, 1stC %d, 1stB %d ",n1stNeuron,n1stCell,n1stBias);
            l->print();
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O = LOAD(outvals +l->iI +i);
                IN = ADD(IN, MUL(O, LOAD(weights +l->iW  +n*l->nI +i)));
                IG = ADD(IG, MUL(O, LOAD(weights +l->iWI +n*l->nI +i)));
                FG = ADD(FG, MUL(O, LOAD(weights +l->iWF +n*l->nI +i)));
                OG = ADD(OG, MUL(O, LOAD(weights +l->iWO +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(outvals +l->iI +i);
                tC += oVal * *(weights +l->iW  +n*l->nI +i);
                tI += oVal * *(weights +l->iWI +n*l->nI +i);
                tF += oVal * *(weights +l->iWF +n*l->nI +i);
                tO += oVal * *(weights +l->iWO +n*l->nI +i);
            }
            #endif
        }
        {
            const Link* const l = recurrent_links;
            printf("Normal prev_input_links with 1stN %d, 1stC %d, 1stB %d ",n1stNeuron,n1stCell,n1stBias);
            l->print();
            #ifdef SIMDKERNELSIN
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O = LOAD(M->outvals +l->iI +i);
                IN = ADD(IN, MUL(O, LOAD(weights +l->iW  +n*l->nI +i)));
                IG = ADD(IG, MUL(O, LOAD(weights +l->iWI +n*l->nI +i)));
                FG = ADD(FG, MUL(O, LOAD(weights +l->iWF +n*l->nI +i)));
                OG = ADD(OG, MUL(O, LOAD(weights +l->iWO +n*l->nI +i)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(M->outvals +l->iI +i);
                tC += oVal * *(weights +l->iW  +n*l->nI +i);
                tI += oVal * *(weights +l->iWI +n*l->nI +i);
                tF += oVal * *(weights +l->iWF +n*l->nI +i);
                tO += oVal * *(weights +l->iWO +n*l->nI +i);
            }
            #endif
        }
        
        #ifdef SIMDKERNELSIN
        STORE (tCry,IN); STORE (tIry,IG); STORE (tFry,FG); STORE (tOry,OG);
        for(int t=0; t<SIMD; t++) {
            tC += *(tCry+t); tI += *(tIry+t); tF += *(tFry+t); tO += *(tOry+t);
        }
        #endif
        tC += *(biases +n1stBias +n);   *(N->in_vals +n1stNeuron +n) = tC;
        tI += *(biases +n1stBiasIG +n); *(N->iIGates +n1stCell   +n) = tI;
        //*(oldstates +n1stCell +n)* *(weights +n1stPeep +3*n)   + ;
        tF += *(biases +n1stBiasFG +n); *(N->iFGates +n1stCell   +n) = tF;
        //*(oldstates +n1stCell +n)* *(weights +n1stPeep +3*n +1)+
        tO += *(biases +n1stBiasOG +n); *(N->iOGates +n1stCell   +n) = tO;
        //*(N->ostates +n1stCell +n) * *(weights +n1stPeep +3*n +2) +
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
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        {
            const Link* const l = output_links;
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(cC->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                           *(cC->errvals +l->iO +i) * (
                                        *(cC->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                                        *(cC->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                                        *(cC->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err  += *(cC->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
        }
        *(C->errvals +n1stNeuron +n) = err * func->evalDiff(*(cC->in_vals +n1stNeuron+n));
    }
}

void NormalLayer::backPropagateDeltaFirst(Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        {
            const Link* const l = output_links;
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(cC->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                           *(cC->errvals +l->iO +i) * (
                                       *(cC->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                                       *(cC->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                                       *(cC->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err  += *(cC->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
        }
        {
            const Link* const l = recurrent_links;
            for (int i=0; i<l->nO; i++)
                err  += *(N->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
        }
        *(C->errvals +n1stNeuron +n) = err * func->evalDiff(*(cC->in_vals +n1stNeuron+n));
    }
}

void NormalLayer::backPropagateDeltaLast(const Lab* const P, Lab* const C, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    for (int n=0; n<nNeurons; n++)  {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        {
            const Link* const l = output_links;
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(cC->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                           *(cC->errvals +l->iO +i) * (
                                        *(cC->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                                        *(cC->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                                        *(cC->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err  += *(cC->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
        }
        *(C->errvals +n1stNeuron +n) = err * func->evalDiff(*(cC->in_vals +n1stNeuron+n));
    }
}

void NormalLayer::backPropagateDelta(const Lab* const P, Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        {
            const Link* const l = output_links;
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(cC->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                           *(cC->errvals +l->iO +i) * (
                                       *(cC->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                                       *(cC->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                                       *(cC->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err  += *(cC->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
        }
        {
            const Link* const l = recurrent_links;
            for (int i=0; i<l->nO; i++)
                    err  += *(N->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
        }
        *(C->errvals +n1stNeuron +n) = err * func->evalDiff(*(cC->in_vals +n1stNeuron+n));
    }
}

void LSTMLayer::backPropagateDeltaFirst(Lab* const C, const Lab* const N, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        {
            const Link* const l = output_links;
            printf("First delta curr_input_links with 1stN %d, 1stC %d, 1stB %d",n1stNeuron,n1stCell,n1stBias);
            l->print();
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(cC->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                           *(cC->errvals +l->iO +i) * (
                           *(cC->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                           *(cC->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                           *(cC->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err += *(cC->errvals +l->iO +i) * *(weights +l->iW  +i*l->nI +n);
        }
        {
            const Link* const l = recurrent_links;
            printf("First delta next_output_links with 1stN %d, 1stC %d, 1stB %d",n1stNeuron,n1stCell,n1stBias);
            l->print();
            for (int i=0; i<l->nO; i++)
                err += *(N->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                       *(N->errvals +l->iO +i) * (
                       *(N->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                       *(N->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                       *(N->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
        }
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
        //the final boss:
        *(C->errvals+n1stNeuron+n) = err * *(cC->oOGates+n1stCell+n) * func->evalDiff(*(cC->ostates +n1stCell +n)) + *(N->errvals+n1stNeuron+n)* *(N->oFGates+n1stCell+n);// +
        //*(next->eIGates+n1stCell+n)* *(weights+n1stPeep+3*n)   +
        //*(next->eFGates+n1stCell+n)* *(weights+n1stPeep+3*n+1) +
         //*(curr->eOGates+n1stCell+n)* *(weights+n1stPeep+3*n+2);
    }
}

void LSTMLayer::backPropagateDelta(Lab* const C, const Real* const weights, const Real* const biases) const
{
    const Lab* const cC = C;
    for (int n=0; n<nNeurons; n++)
    {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        {
            const Link* const l = output_links;
            printf("Solo delta curr_input_links with 1stN %d, 1stC %d, 1stB %d",n1stNeuron,n1stCell,n1stBias);
            l->print();
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(cC->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                           *(cC->errvals +l->iO +i) * (
                                       *(cC->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                                       *(cC->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                                       *(cC->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err  += *(cC->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
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
    
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        
        {
            const Link* const l = output_links;
            printf("Normal delta curr_input_links with 1stN %d, 1stC %d, 1stB %d",n1stNeuron,n1stCell,n1stBias);
            l->print();
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(cC->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                           *(cC->errvals +l->iO +i) * (
                           *(cC->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                           *(cC->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                           *(cC->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err += *(cC->errvals +l->iO +i) * *(weights +l->iW  +i*l->nI +n);
        }
        {
            const Link* const l = recurrent_links;
            printf("Normal delta next_output_links with 1stN %d, 1stC %d, 1stB %d",n1stNeuron,n1stCell,n1stBias);
            l->print();
            for (int i=0; i<l->nO; i++)
                err += *(N->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                       *(N->errvals +l->iO +i) * (
                       *(N->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                       *(N->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                       *(N->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
        }
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
    for (int n=0; n<nNeurons; n++) {
        Real err = (last) ? *(cC->errvals +n1stNeuron +n) : 0.0;
        
        {
            const Link* const l = output_links;
            printf("Last delta curr_input_links with 1stN %d, 1stC %d, 1stB %d",n1stNeuron,n1stCell,n1stBias);
            l->print();
            if (l->LSTM)
                for (int i=0; i<l->nO; i++)
                    err += *(cC->eOGates +l->iC +i) * *(weights +l->iWO +i*l->nI +n) +
                           *(cC->errvals +l->iO +i) * (
                               *(cC->eMCell  +l->iC +i) * *(weights +l->iW  +i*l->nI +n) +
                               *(cC->eIGates +l->iC +i) * *(weights +l->iWI +i*l->nI +n) +
                               *(cC->eFGates +l->iC +i) * *(weights +l->iWF +i*l->nI +n) );
            else
                for (int i=0; i<l->nO; i++)
                    err  += *(cC->errvals +l->iO +i) * *(weights +l->iW +i*l->nI +n);
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
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) = eC;
        {
            const Link* const l = input_links;
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
        {
            const Link* const l = recurrent_links;
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

void LSTMLayer::backPropagateGrads(const Lab* const P, const Lab* const C, Grads* const grad) const
{
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) = eC; *(grad->_B +n1stBiasIG +n) = eI;
        *(grad->_B +n1stBiasFG +n) = eF; *(grad->_B +n1stBiasOG +n) = eO;
        
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSG
            const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
            const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O  = LOAD(C->outvals +l->iI +i);
                STORE(grad->_W +l->iW  +n*l->nI +i, MUL(EN,O));
                STORE(grad->_W +l->iWI +n*l->nI +i, MUL(EI,O));
                STORE(grad->_W +l->iWF +n*l->nI +i, MUL(EF,O));
                STORE(grad->_W +l->iWO +n*l->nI +i, MUL(EO,O));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(C->outvals+l->iI+i);
                *(grad->_W+l->iW +n*l->nI+i) = oVal * eC;
                *(grad->_W+l->iWI+n*l->nI+i) = oVal * eI;
                *(grad->_W+l->iWF+n*l->nI+i) = oVal * eF;
                *(grad->_W+l->iWO+n*l->nI+i) = oVal * eO;
            }
            #endif
        }
        {
            const Link* const l = recurrent_links;
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

void NormalLayer::backPropagateGrads(const Lab* const C, Grads* const grad) const
{
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) = eC;
        {
            const Link* const l = input_links;
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
    }
}

void LSTMLayer::backPropagateGrads(const Lab* const C, Grads* const grad) const
{
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) = eC; *(grad->_B +n1stBiasIG +n) = eI;
        *(grad->_B +n1stBiasFG +n) = eF; *(grad->_B +n1stBiasOG +n) = eO;
        
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSG
            const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
            const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O  = LOAD(C->outvals +l->iI +i);
                STORE(grad->_W +l->iW  +n*l->nI +i, MUL(EN,O));
                STORE(grad->_W +l->iWI +n*l->nI +i, MUL(EI,O));
                STORE(grad->_W +l->iWF +n*l->nI +i, MUL(EF,O));
                STORE(grad->_W +l->iWO +n*l->nI +i, MUL(EO,O));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(C->outvals+l->iI+i);
                *(grad->_W+l->iW +n*l->nI+i) = oVal * eC;
                *(grad->_W+l->iWI+n*l->nI+i) = oVal * eI;
                *(grad->_W+l->iWF+n*l->nI+i) = oVal * eF;
                *(grad->_W+l->iWO+n*l->nI+i) = oVal * eO;
            }
            #endif
        }
    }
}

void NormalLayer::backPropagateAddGrads(const Lab* const P, const Lab* const C, Grads* const grad) const
{
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) += eC;
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSG
            const vec E = BCAST(&eC);
            for (int i=0; i<l->nI; i+=SIMD) {
                STORE(grad->_W +l->iW +n*l->nI +i,
                      ADD(LOAD(grad->_W +l->iW +n*l->nI +i),
                          MUL(E, LOAD(C->outvals +l->iI +i))));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                *(grad->_W +l->iW +n*l->nI +i) += *(C->outvals +l->iI +i) * eC;
            }
            #endif
        }
        {
            const Link* const l = recurrent_links;
            #ifdef SIMDKERNELSG
            const vec E = BCAST(&eC);
            for (int i=0; i<l->nI; i+=SIMD) {
                STORE(grad->_W +l->iW +n*l->nI +i,
                      ADD(LOAD(grad->_W +l->iW +n*l->nI +i),
                          MUL(E, LOAD(P->outvals +l->iI +i))));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                *(grad->_W +l->iW +n*l->nI +i) += *(P->outvals +l->iI +i) * eC;
            }
            #endif
        }
    }
}

void LSTMLayer::backPropagateAddGrads(const Lab* const P, const Lab* const C, Grads* const grad) const
{
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) += eC; *(grad->_B +n1stBiasIG +n) += eI;
        *(grad->_B +n1stBiasFG +n) += eF; *(grad->_B +n1stBiasOG +n) += eO;
        
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSG
            const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
            const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O  = LOAD(C->outvals +l->iI +i);
                
                   STORE(grad->_W +l->iW  +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iW  +n*l->nI +i), MUL(EN,O)));
                   STORE(grad->_W +l->iWI +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iWI +n*l->nI +i), MUL(EI,O)));
                   STORE(grad->_W +l->iWF +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iWF +n*l->nI +i), MUL(EF,O)));
                   STORE(grad->_W +l->iWO +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iWO +n*l->nI +i), MUL(EO,O)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(C->outvals+l->iI+i);
                *(grad->_W+l->iW +n*l->nI+i) += oVal * eC;
                *(grad->_W+l->iWI+n*l->nI+i) += oVal * eI;
                *(grad->_W+l->iWF+n*l->nI+i) += oVal * eF;
                *(grad->_W+l->iWO+n*l->nI+i) += oVal * eO;
            }
            #endif
        }
        {
            const Link* const l = recurrent_links;
            #ifdef SIMDKERNELSG
            const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
            const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O  = LOAD(P->outvals +l->iI +i);
                   STORE(grad->_W +l->iW  +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iW  +n*l->nI +i), MUL(EN,O)));
                   STORE(grad->_W +l->iWI +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iWI +n*l->nI +i), MUL(EI,O)));
                   STORE(grad->_W +l->iWF +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iWF +n*l->nI +i), MUL(EF,O)));
                   STORE(grad->_W +l->iWO +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iWO +n*l->nI +i), MUL(EO,O)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(P->outvals+l->iI+i);
                *(grad->_W+l->iW +n*l->nI+i) += oVal * eC;
                *(grad->_W+l->iWI+n*l->nI+i) += oVal * eI;
                *(grad->_W+l->iWF+n*l->nI+i) += oVal * eF;
                *(grad->_W+l->iWO+n*l->nI+i) += oVal * eO;
            }
            #endif
        }
    }
}

void NormalLayer::backPropagateAddGrads(const Lab* const C, Grads* const grad) const
{
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->errvals +n1stNeuron +n);
        *(grad->_B +n1stBias +n) = eC;
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSG
            const vec E = BCAST(&eC);
            for (int i=0; i<l->nI; i+=SIMD) {
                        STORE(grad->_W +l->iW +n*l->nI +i,
                     ADD(LOAD(grad->_W +l->iW +n*l->nI +i),
                          MUL(E, LOAD(C->outvals +l->iI +i))));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                *(grad->_W +l->iW +n*l->nI +i) += *(C->outvals +l->iI +i) * eC;
            }
            #endif
        }
    }
}

void LSTMLayer::backPropagateAddGrads(const Lab* const C, Grads* const grad) const
{
    for (int n=0; n<nNeurons; n++) {
        const Real eC = *(C->eMCell  +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eI = *(C->eIGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eF = *(C->eFGates +n1stCell +n) * *(C->errvals +n1stNeuron +n);
        const Real eO = *(C->eOGates +n1stCell +n);
        
        *(grad->_B +n1stBias   +n) = eC; *(grad->_B +n1stBiasIG +n) = eI;
        *(grad->_B +n1stBiasFG +n) = eF; *(grad->_B +n1stBiasOG +n) = eO;
        
        {
            const Link* const l = input_links;
            #ifdef SIMDKERNELSG
            const vec EN = BCAST(&eC); const vec EI = BCAST(&eI);
            const vec EF = BCAST(&eF); const vec EO = BCAST(&eO);
            for (int i=0; i<l->nI; i+=SIMD) {
                const vec O  = LOAD(C->outvals +l->iI +i);
                   STORE(grad->_W +l->iW  +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iW  +n*l->nI +i), MUL(EN,O)));
                   STORE(grad->_W +l->iWI +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iWI +n*l->nI +i), MUL(EI,O)));
                   STORE(grad->_W +l->iWF +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iWF +n*l->nI +i), MUL(EF,O)));
                   STORE(grad->_W +l->iWO +n*l->nI +i,
                ADD(LOAD(grad->_W +l->iWO +n*l->nI +i), MUL(EO,O)));
            }
            #else
            for (int i=0; i<l->nI; i++) {
                const Real oVal = *(C->outvals+l->iI+i);
                *(grad->_W+l->iW +n*l->nI+i) += oVal * eC;
                *(grad->_W+l->iWI+n*l->nI+i) += oVal * eI;
                *(grad->_W+l->iWF+n*l->nI+i) += oVal * eF;
                *(grad->_W+l->iWO+n*l->nI+i) += oVal * eO;
            }
            #endif
        }
    }
}