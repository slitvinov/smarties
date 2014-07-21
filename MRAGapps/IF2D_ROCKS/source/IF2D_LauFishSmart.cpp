/*
 * IF2D_LauFishSmart.cpp
 *
 *  Created on: Jun 10, 2014
 *      From  : mgazzola
 */

#include "IF2D_LauFishSmart.h"
#include <stdio.h>
#include <math.h>
#include <fstream> 

IF2D_LauFishSmart::IF2D_LauFishSmart(ArgumentParser & parser, Grid<W,B>& grid, const Real _xm, const Real _ym, const Real D, const Real T, const Real tau, const Real angle, const Real eps, const Real Uinf[2], IF2D_PenalizationOperator& penalization, const int LMAX, const int ID, RL::RL_TabularPolicy ** _policy, const int seed) :
        IF2D_StefanFish(parser, grid, _xm, _ym, D, T, 0.0, tau, angle, vector<double>(6), vector<double>(6), 0.0, eps, Uinf, penalization, LMAX, ID, _policy, seed)
{
    myshape = static_cast<StefanFish*>(shape);

    posTarget[0] = 0.5; // not used for now
    posTarget[1] = 0.5;

    printf(" --- LauFishSmart --- \n");
}

IF2D_LauFishSmart::~IF2D_LauFishSmart()
{
}

namespace LauFish
{
struct FillBlocks
{
    double eps;
    IF2D_CarlingFish::Fish * shape;

    FillBlocks(double eps, IF2D_CarlingFish::Fish *_shape): eps(eps) { shape = _shape; }

    FillBlocks(const FillBlocks& c): eps(c.eps) { shape = c.shape; }

    static bool _is_touching(double eps, const IF2D_CarlingFish::Fish * fish, const BlockInfo& info)
    {
        double min_pos[2], max_pos[2];

        info.pos(min_pos, 0,0);
        info.pos(max_pos, FluidBlock2D::sizeX-1, FluidBlock2D::sizeY-1);

        double bbox[2][2];
        fish->bbox(eps, bbox[0], bbox[1]);

        double intersection[2][2] = {
                max(min_pos[0], bbox[0][0]), min(max_pos[0], bbox[1][0]),
                max(min_pos[1], bbox[0][1]), min(max_pos[1], bbox[1][1]),
        };

        return
                intersection[0][1]-intersection[0][0]>0 &&
                intersection[1][1]-intersection[1][0]>0 ;
    }

    inline void operator()(const BlockInfo& info, FluidBlock2D& b) const
    {
        bool bEmpty = true;

        if(_is_touching(eps, shape, info))
        {
            for(int iy=0; iy<FluidBlock2D::sizeY; iy++)
                for(int ix=0; ix<FluidBlock2D::sizeX; ix++)
                {
                    double p[2];
                    info.pos(p, ix, iy);
                    b(ix,iy).tmp = max( (Real)shape->sample(p[0], p[1]), b(ix,iy).tmp );
                }

            bEmpty = false;
        }
    }
};



struct ComputeAll
{
    Real Uinf[2];
    double eps, xcm, ycm;
    IF2D_CarlingFish::Fish * shape;
    map<int, vector<double> >& b2sum;
    map<int, bool>& b2nonempty;

    ComputeAll(double xcm, double ycm, double eps, IF2D_CarlingFish::Fish *_shape, map<int, vector<double> >& b2sum, Real Uinf_[2], map<int, bool>& b2nonempty): xcm(xcm), ycm(ycm), eps(eps), b2sum(b2sum), b2nonempty(b2nonempty)
    {
        this->Uinf[0] = Uinf_[0];
        this->Uinf[1] = Uinf_[1];

        shape = _shape;
    }

    ComputeAll(const ComputeAll& c): xcm(c.xcm), ycm(c.ycm), eps(c.eps), b2sum(c.b2sum), b2nonempty(c.b2nonempty)
    {
        Uinf[0] = c.Uinf[0];
        Uinf[1] = c.Uinf[1];

        shape = c.shape;
    }

    inline void operator()(const BlockInfo& info, FluidBlock2D& b) const
    {
        bool bNonEmpty = false;

        if(FillBlocks::_is_touching(eps, shape, info))
        {
            double mass = 0;
            double J = 0;
            double vxbar = 0;
            double vybar = 0;
            double omegabar = 0;

            for(int iy=0; iy<FluidBlock2D::sizeY; iy++)
                for(int ix=0; ix<FluidBlock2D::sizeX; ix++)
                {
                    double p[2];
                    info.pos(p, ix, iy);

                    const double Xs =shape->SHARP?  shape->sample(p[0], p[1], info.h[0]) : shape->sample(p[0], p[1]);
                    bNonEmpty |= Xs>0;

                    mass += Xs;
                    J += Xs*((p[0]-xcm)*(p[0]-xcm) + (p[1]-ycm)*(p[1]-ycm));
                    vxbar += Xs*(b(ix, iy).u[0]+Uinf[0]);
                    vybar += Xs*(b(ix, iy).u[1]+Uinf[1]);
                    omegabar += Xs*((b(ix, iy).u[1]+Uinf[1])*(p[0]-xcm)-(b(ix, iy).u[0]+Uinf[0])*(p[1]-ycm));
                }

            assert(b2sum.find(info.blockID) != b2sum.end());
            assert(b2nonempty.find(info.blockID) != b2nonempty.end());

            b2sum[info.blockID][0] = mass*info.h[0]*info.h[0];
            b2sum[info.blockID][1] = J*info.h[0]*info.h[0];
            b2sum[info.blockID][2] = vxbar*info.h[0]*info.h[0];
            b2sum[info.blockID][3] = vybar*info.h[0]*info.h[0];
            b2sum[info.blockID][4] = omegabar*info.h[0]*info.h[0];
            b2nonempty[info.blockID] = bNonEmpty;

        }
    }
};


struct FillVelblocks
{
    double eps, xcm, ycm, vxcorr, vycorr, omegacorr;
    IF2D_CarlingFish::Fish * shape;
    vector<pair< BlockInfo, VelocityBlock *> >& workitems;

    FillVelblocks(vector<pair< BlockInfo, VelocityBlock *> >& workitems, double xcm, double ycm, double vxcorr, double vycorr, double omegacorr, double eps, IF2D_CarlingFish::Fish *_shape):
        workitems(workitems), xcm(xcm), ycm(ycm), vxcorr(vxcorr), vycorr(vycorr), omegacorr(omegacorr), eps(eps)
    {
        shape = _shape;
    }

    FillVelblocks(const FillVelblocks& c): workitems(c.workitems), xcm(c.xcm), ycm(c.ycm), vxcorr(c.vxcorr), vycorr(c.vycorr), omegacorr(c.omegacorr), eps(c.eps)
    {
        shape = c.shape;
    }

    inline void operator()(blocked_range<int> range) const
    {
        for(int iblock=range.begin(); iblock<range.end(); iblock++)
        {
            BlockInfo info = workitems[iblock].first;
            VelocityBlock * u_desired = workitems[iblock].second;

            const double xm = xcm;
            const double ym = ycm;
            const double av = shape->angular_velocity - omegacorr;
            const double vx = shape->vx - vxcorr;
            const double vy = shape->vy - vycorr;

            if(FillBlocks::_is_touching(eps, shape, info))
                for(int iy=0; iy<FluidBlock2D::sizeY; iy++)
                    for(int ix=0; ix<FluidBlock2D::sizeX; ix++)
                    {
                        double p[2];
                        info.pos(p, ix, iy);

                        double Xs = 0.0;
                        double vdefx = 0.0;
                        double vdefy = 0.0;
                        if (shape->SHARP)
                          shape->sample(p[0], p[1], Xs, vdefx, vdefy, info.h[0]);
                        else
                          shape->sample(p[0], p[1], Xs, vdefx, vdefy);

                        if (Xs > 0)
                        {
                            u_desired->u[0][iy][ix] = - av*(p[1]-ym) + vx + vdefx;
                            u_desired->u[1][iy][ix] = + av*(p[0]-xm) + vy + vdefy;
                        }
                        else
                        {
                            u_desired->u[0][iy][ix] = 0.0;
                            u_desired->u[1][iy][ix] = 0.0;
                        }
                    }
        }
    }
};
}

double IF2D_LauFishSmart::getState()
{
  const double state = shape->ym - posTarget[1];
  return state;
}

double IF2D_LauFishSmart::getReward()
{
    const double reward = - fabs(shape->ym - posTarget[1]);
    return reward;    
}

void IF2D_LauFishSmart::act(const int action)
{
    const int sizeCurvature = myshape->CURVATURE.size();
    const int sizeBaseline  = myshape->CURVATURE.size();
    assert(sizeCurvature==6);
    assert(sizeBaseline==6);

    myshape->mTransitionTime = 1;
    myshape->tau = 1.44;
    myshape->CURVATURE[0] = 0.0;
    myshape->CURVATURE[1] = 1.51/3.0;
    myshape->CURVATURE[2] = 0.48/3.0;
    myshape->CURVATURE[3] = 5.74/3.0;
    myshape->CURVATURE[4] = 2.73/3.0;
    myshape->CURVATURE[5] = 0.0;
    switch(action)
    {
    case 0 :
        printf("Go straight!!\n");
        for(int i = 0; i < sizeBaseline; i++)
            myshape->BASELINE[i]=0.0;
        movState = STRAIGHT;
        break;
    case 1 :
        printf("Turn left!!\n");
        for(int i = 0; i < sizeBaseline; i++)
            myshape->BASELINE[i]=-1.0;
        movState = TURN;
        break;
    case 2 :
        printf("Turn right!!\n");
        for(int i = 0; i < sizeBaseline; i++)
            myshape->BASELINE[i]=1.0;
        movState = TURN;
        break;
    default :
        printf("Error on the action!!\n");
        break;
    }
}


// void IF2D_LauFishSmart::saveCurvature(double &t, int  &nbReset)
// {
//     const int sizeBaseline  = myshape->CURVATURE.size();
//
//     FILE* fkd;
//     fkd = fopen("FishCurrentShape.txt", "a");
//     fprintf(fkd, "%10.6f %3i %15.5e %15.5e", t, nbReset, myshape->mTCurrent, myshape->mTauCurrent);
//     for(int i = 0; i < sizeBaseline; i++)
//         fprintf(fkd, "%15.5e", myshape->mCurrentCurvature[i]);
//     for(int i = 0; i < sizeBaseline; i++)
//         fprintf(fkd, "%15.5e", myshape->mCurrentBaseline[i]);
//     fprintf(fkd, "\n");
//     fclose(fkd);
// }


void IF2D_LauFishSmart::adaptVelocity(const double t, double Uinf_in[2])
{
    FILE * f = fopen("tmpVelocity.txt", "a");
    if (f!=NULL)
    {
        fprintf(f,"%f\t%e\t%e\n", t, this->Uinf[0], shape->vx );
        fclose(f);
    }

    this->Uinf[0] = 0;
    Uinf_in[0] = -shape->vx;
    shape->vx=0;
    fillVelocity();
}

void IF2D_LauFishSmart::computeDesiredVelocity(const double t)
{
    const int NQUANTITIES = 5;

    double mass = 0.0;
    double J = 0.0;
    double xbar = 0.0;
    double ybar = 0.0;
    double vxbar = 0.0;
    double vybar = 0.0;
    double omegabar = 0.0;

    vInfo = grid.getBlocksInfo();
    const BlockCollection<B>& coll = grid.getBlockCollection();

    map<int, vector<double> > integrals;
    for(vector<BlockInfo>::const_iterator it=vInfo.begin(); it!=vInfo.end(); it++)
        integrals[it->blockID] = vector<double>(NQUANTITIES);

    nonempty.clear();
    for(vector<BlockInfo>::const_iterator it=vInfo.begin(); it!=vInfo.end(); it++)
        nonempty[it->blockID] = false;

    // Compute all from the flow
    mass = J = xbar = ybar = vxbar = vybar = omegabar = 0.0;
    LauFish::ComputeAll computeAll(xm_corr, ym_corr, eps, shape, integrals, Uinf, nonempty);
    block_processing.process(vInfo, coll, computeAll);
    for(map<int, vector< double> >::const_iterator it= integrals.begin(); it!=integrals.end(); ++it)
    {
        mass += (it->second)[0];
        J += (it->second)[1];
        vxbar += (it->second)[2];
        vybar += (it->second)[3];
        omegabar += (it->second)[4];            
    }
    vxbar /= mass;
    vybar /= mass;
    omegabar /= J;

    // Set the right vx, vy and angular velocity
    shape->vx = vxbar;
    shape->vy = vybar;
    shape->angular_velocity = omegabar;
    shape->m = mass;
    shape->J = J;

    fillVelocity();
}

void IF2D_LauFishSmart::fillVelocity()
{
    // Prepare desired velocity blocks
    for (map<int, const VelocityBlock *>::iterator it = desired_velocity.begin(); it!= desired_velocity.end(); it++)
    {
        assert(it->second != NULL);
        VelocityBlock::deallocate(it->second);
    }
    desired_velocity.clear();

    vector<pair< BlockInfo, VelocityBlock *> > velblocks;
    for(vector<BlockInfo>::const_iterator it=vInfo.begin(); it!=vInfo.end(); it++)
    {
        if(nonempty[it->blockID] == true)
        {
            VelocityBlock * velblock = VelocityBlock::allocate(1);
            desired_velocity[it->blockID] = velblock;
            velblocks.push_back(pair< BlockInfo, VelocityBlock *>(*it, velblock));
        }
    }

    // Set desired velocities
    LauFish::FillVelblocks fillvelblocks(velblocks, xm_corr, ym_corr, vx_corr, vy_corr, omega_corr, eps, shape);
    tbb::parallel_for(blocked_range<int>(0, velblocks.size()), fillvelblocks, auto_partitioner());

    penalization.set_desired_velocity(&desired_velocity);
}
