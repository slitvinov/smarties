/*
 * IF2D_LauFishSmart.h
 *
 *  Created on: Jun 10, 2014
 *      Author: laurentm
 *      From  : mgazzola
 */

#ifndef IF2D_LAUFISHSMART_H_
#define IF2D_LAUFISHSMART_H_

#include "IF2D_Headers.h"
#include "IF2D_Types.h"
#include "IF2D_StefanFish.h" 

enum MovingStates {STRAIGHT, TURN};

class IF2D_LauFishSmart : public IF2D_StefanFish
{
 protected:
    bool inside;
    vector<BlockInfo> vInfo;
    StefanFish* myshape;
	map<int, bool> nonempty;
	double posTarget[2];
    void fillVelocity();

 public:
    IF2D_LauFishSmart(ArgumentParser & parser, Grid<W,B>& grid, const Real _xm, const Real _ym, const Real D, const Real T, const Real tau, const Real angle, const Real eps, const Real Uinf[2], IF2D_PenalizationOperator& penalization, const int LMAX, const int ID = 0, RL::RL_TabularPolicy ** _policy = NULL, const int seed = 0);
    virtual ~IF2D_LauFishSmart();

    MovingStates movState;

    virtual double getState();
    virtual double getReward();
    virtual void   act(const int action);
    // virtual void   saveCurvature(double &t, int  &nbReset);
    void adaptVelocity(const double t, double Uinf_in[2]);
    void computeDesiredVelocity(const double t);
    void restartcochon(const double t, string filename = std::string());

    virtual void set_Uinfinity(const Real Uinf[2])
    {
        this->Uinf[0] = Uinf[0];
        this->Uinf[1] = Uinf[1];
    }
};

#endif /* IF2D_LAUFISHSMART_H_ */
