/*
 *  untitled.h
 *  IF2D_ROCKS
 *
 *  Created by Chloe Mimeau on 4/1/11.
 *  Copyright 2011 ETHZ. All rights reserved.
 *
 */

#include "IF2D_FluidMediatedLau.h"
#include "IF2D_AdvectionOperator_Particles.h"
#include "IF2D_VelocitySolver_Mani.h"
#include "IF2D_VelocitySolver_Wim.h" // FPFiO
#include "IF2D_PotentialSolver_Mattia.h"

// #include "IF2D_FloatingObstacleVector.h"
#include <limits>
#include "IF2D_Clear.h"

#ifdef _IF2D_MPI_
#include "IF2D_VelocitySolverMPI_Mani.h"
#endif

#include "IF2D_LauFishSmart.h"

static const int maxParticleStencil[2][3] = {
        -3, -3, 0,
        +4, +4, +1
};

IF2D_FluidMediatedLau::IF2D_FluidMediatedLau(const int argc, const char ** argv): 
                                                parser(argc, argv), t(0), step_id(0),
                                                velsolver(NULL),penalization(NULL), advection(NULL), diffusion(NULL),bReset(false)
{
    printf("////////////////////////////////////////////////////////////\n");
    printf("////////////            AVE MARIA         ///////////////\n");
    printf("////////////////////////////////////////////////////////////\n");

    bRESTART = parser("-restart").asBool();
    sOBSTACLE = parser("-obstacle").asString();
    sRIGID_INLET_TYPE = parser("-rio").asString();
    bREFINEOMEGAONLY = parser("-refine-omega-only").asBool();
    bFMMSKIP = parser("-fmm-skip").asBool();
    LAMBDADT = parser("-lambdadt").asDouble();
    XPOS = parser("-xpos").asDouble();
    YPOS = parser("-ypos").asDouble();

    parser.set_strict_mode();

    BPD = parser("-bpd").asInt();
    JUMP = parser("-jump").asInt();
    LMAX = parser("-lmax").asInt();
    TEND = parser("-tend").asDouble();
    RAMP = parser("-ramp").asInt();
    ADAPTFREQ = parser("-adaptfreq").asInt();
    DUMPFREQ = parser("-dumpfreq").asDouble();
    SAVEFREQ = parser("-savefreq").asInt();
    RE = parser("-re").asDouble();
    CFL = parser("-cfl").asDouble();
    LCFL = parser("-lcfl").asDouble();
    RTOL = parser("-rtol").asDouble();
    CTOL = parser("-ctol").asDouble(); 
    LAMBDA = parser("-lambda").asDouble();
    D = parser("-D").asDouble();
    bPARTICLES = parser("-particles").asBool();
    bUNIFORM = parser("-uniform").asBool();
    sFMMSOLVER = parser("-fmm").asString();
    MOLLFACTOR = parser("-mollfactor").asInt();
    Uinf[0] = parser("-uinfx").asDouble();
    Uinf[1] = parser("-uinfy").asDouble();
    FC = parser("-fc").asDouble();
    const bool HILBERT = parser("-hilbert").asBool();

    // LauFish
    bADAPTVEL = parser("-adaptvel").asBool();
    bLEARNING= parser("-learning").asBool();
    nbObstacle = 1;
    d = parser("-d").asDouble();
    xm = parser("-xm").asDouble();
    ym = parser("-ym").asDouble();
    tau = parser("-tau").asDouble();
    angle = parser("-angle").asDouble();
    T  = parser("-T").asDouble();

    charLength = D;
    charVel = D/T;
    d = d*charLength;
    xm = XPOS + xm*charLength;
    ym = YPOS + ym*charLength;
    angle = angle/180.0*M_PI;
    
    // Learning 
    nbReset = 0;

    parser.save_options();

    assert(TEND >= 0.0);
    assert(BPD > 1);
    assert(ADAPTFREQ > 0);
    assert(DUMPFREQ >= 0);
    assert(JUMP >= 1);
    assert(LMAX >= 0);
    assert(RE > 0);
    assert(CFL > 0 && CFL<1);
    assert(LCFL > 0 && LCFL<1);
    assert(RTOL > 0);
    assert(CTOL > 0);
    assert(LAMBDA > 0);
    assert(sFMMSOLVER != "");
    assert(sRIGID_INLET_TYPE != "");
    assert(MOLLFACTOR > 0);
    assert(FC>0.0 && FC<1.0);

#ifdef _IF2D_MPI_
    if (sFMMSOLVER == "mpi-velocity")
        velsolver = new IF2D_VelocitySolverMPI_Mani(argc, argv);
#endif

    const Real h_min = 1./FluidBlock2D::sizeX*pow(0.5, LMAX);

    Real pos[2] = {0.5,0.5};
    if(XPOS>0 && XPOS<1.0){ pos[0] = XPOS; }
    if(YPOS>0 && YPOS<1.0){ pos[1] = YPOS; }

    if (HILBERT)
    {
        if (bPARTICLES)
            grid = new Grid_Hilbert2D<W,B>(BPD,BPD,1, maxParticleStencil);
        else
            grid = new Grid_Hilbert2D<W,B>(BPD,BPD,1);
    }
    else
    {
        if (bPARTICLES)
            grid = new Grid<W,B>(BPD,BPD,1, maxParticleStencil);
        else
            grid = new Grid<W,B>(BPD,BPD,1);
    }


    assert(grid != NULL);

    refiner = new Refiner_BlackList(JUMP, LMAX);
    compressor = new Compressor(JUMP);
    grid->setRefiner(refiner);
    grid->setCompressor(compressor);

    //const Real h_spaceconv = 1./FluidBlock2D::sizeX*pow(0.5, 4);
    //epsilon = (Real)MOLLFACTOR*sqrt(2.)*h_spaceconv;

    epsilon = (Real)MOLLFACTOR*sqrt(2.)*h_min;

    penalization = new IF2D_PenalizationOperator(*grid, LAMBDA, Uinf, bRESTART);

    if (bPARTICLES)
        advection =new IF2D_AdvectionOperator_Particles(*grid, CFL, LCFL);
    else
        advection =new IF2D_AdvectionOperator(*grid, CFL);

    advection->set_Uinfinity(Uinf);


    if(sFMMSOLVER == "velocity")
        velsolver = new IF2D_VelocitySolver_Mani(*grid, parser);
    else if(sFMMSOLVER == "velocity-wim")
        velsolver = new IF2D_VelocitySolver_Wim(*grid, parser);
    else 
    {
#ifdef _IF2D_MPI_
        if (sFMMSOLVER == "mpi-velocity")
            ((IF2D_VelocitySolverMPI_Mani *) velsolver)->set_grid(*grid);
#endif
        if(velsolver == NULL)
        {
            printf("VELOCITY SOLVER CANNOT BE NULL AT THIS POINT. aborting...");
            abort();
        }
    }

    diffusion = new IF2D_DiffusionOperator_4thOrder(*grid, nu, FC);

    bUSEPOTENTIAL = parser("-fmm-potential").asBool();

    if(bUSEPOTENTIAL)
      {
        if(sFMMSOLVER == "velocity")
          potsolver = new IF2D_PotentialSolver_Mattia(*grid, parser);
        else
          {
        printf("No potential solver MPI!!\n");
        abort();
          }
      }

    // Extra parsing
    std::string sFACTORY = parser("-factory").asString();
    assert(sFACTORY != "");

    assert( parser("-obstacle").asString() == "heterogeneous" );
    {
        nu = charVel*charLength/RE;
        diffusion->set_viscosity(nu);

        myObstacle = new IF2D_LauFishSmart(parser, *grid, xm, ym, d, T, tau, angle, epsilon, Uinf, *penalization, LMAX, nbObstacle);
    }

    parser.unset_strict_mode();

    // use vorticity killing?
    bUSEKILLVORT = parser("-usekillvort").asBool();

    if(bUSEKILLVORT)
    {
        // initialize vorticity killing stuff
        KILLVORT = parser("-killvort").asInt();
        killVort = new IF2D_KillVortRightBoundaryOperator(*grid, KILLVORT);
    }
    else
    {
        // vorticity killing stuff not used --> defang it
        KILLVORT = 0;
        killVort = NULL;
    }

    // coupled with optimization?
    bUSEOPTIMIZER = parser("-useoptimizer").asBool();
    if(bUSEOPTIMIZER)
    {
        TBOUND = parser("-tbound").asDouble();
        // as long as t<TBOUND the fitness file contains a huge number
        FILE * fitnessFile = fopen("fitness","w");
        assert(fitnessFile!=NULL);
        fprintf(fitnessFile,"%e",std::numeric_limits<Real>::max());
        fclose(fitnessFile);
    }
    else
        TBOUND = 0.0;


    FILE * ppFile = fopen("header.txt","w");
    fprintf(ppFile, "LCFL=%e\n", LCFL);
    fprintf(ppFile, "LMAX=%d\n", LMAX);
    fprintf(ppFile, "RE=%e\n", RE);
    fprintf(ppFile, "charLength=%e\n",charLength);
    fprintf(ppFile, "charVel=%e\n",charVel);
    fprintf(ppFile, "nu=%10.10e\n", nu);
    fflush(ppFile);
    fclose(ppFile);

    if(myObstacle!=NULL)
    {
        if(bRESTART)
        {
            _restart();
            _dump("restartedcondition");
        }
        else
        {
            _ic(*grid); // initial condition: set omega, vx, vy and t to 0
            _refine(true);
            _compress(true);
            _dump("initialcondition");
        }
    }
}

IF2D_FluidMediatedLau::~IF2D_FluidMediatedLau()
{
    if(velsolver!=NULL){ delete velsolver; velsolver=NULL; }
    if(penalization!=NULL){ delete penalization; penalization=NULL; }
    if(advection!=NULL){ delete advection; advection=NULL; }
    if(diffusion!=NULL){ delete diffusion; diffusion=NULL; }

    if(grid!=NULL){ delete grid; grid=NULL; }
    if(refiner!=NULL){ delete refiner; refiner=NULL; }
    if(compressor!=NULL){ delete compressor; compressor=NULL; }

    if(myObstacle!=NULL)
    {
        delete myObstacle;
        myObstacle = NULL;
    }

    if(killVort!=NULL)
    {
        delete killVort;
        killVort = NULL;
    }
}

void IF2D_FluidMediatedLau::_restart()
{   
    //read status
    {
        FILE * f = fopen("restart.status", "r");
        assert(f != NULL);
        float val = -1;
        fscanf(f, "time: %e\n", &val);
        assert(val>=0);
        t=val;
        int step_id_fake = -1;
        fscanf(f, "stepid: %d\n", &step_id_fake);
        step_id = step_id_fake;
        assert(step_id >= 0);
        int nbReset_fake = -1;
        fscanf(f, "nbReset: %d\n", &nbReset_fake);
        nbReset = nbReset_fake;
        assert(nbReset >= 0);
        fclose(f);
    }

    printf("DESERIALIZATION: time is %f and step id is %d\n", t, (int)step_id);

    //read grid
    IO_Binary<W,B> serializer;
    serializer.Read(*grid, "restart");

    char buf[500];
    sprintf(buf, "restart.shape%04d", 0);
    string f(buf);

    myObstacle->restart(t,f);
    myObstacle->refresh(t,f);

    IF2D_Clear cleaner;
    cleaner.clearTmp(*grid);

    myObstacle->create(t);
    myObstacle->computeDesiredVelocity(t);
    if(bADAPTVEL)                         
    {
        myObstacle->adaptVelocity(t,Uinf);
        advection->set_Uinfinity(Uinf);
        penalization->set_Uinfinity(Uinf); 
    }
}

void IF2D_FluidMediatedLau::reset()
{
    nbReset++;
    printf("\n----- Agent outside = RESET counter: %i ----\n", nbReset);

    // kill the obstacle
    if(myObstacle!=NULL)
    {
        delete myObstacle;
        myObstacle=NULL;
    }

    // write some logfile with time where it was restarted
    FILE * f = fopen("resetHistory.txt", "a");
    if (f != NULL)
    {
        fprintf(f, "%d %20.20e %d \n", nbReset, t, (int)step_id);
        fclose(f);
    }

    // reset the time and step_id
    t = 0.0;
    step_id = 0;

    // reinitialize the obstacle
    myObstacle = new IF2D_LauFishSmart(parser, *grid, xm, ym, d, T, tau, angle, epsilon, Uinf, *penalization, LMAX, nbObstacle);

    // reset the grid
    _ic(*grid); 
    _refine(true);
    _compress(true);

    bReset = false;
}

void IF2D_FluidMediatedLau::_save()
{
    IF2D_Clear cleaner;
    cleaner.clearTmp(*grid);
    myObstacle->characteristic_function();

    printf("****SERIALIZING****\n");

    //write status
    {
        FILE * f = fopen("restart.status", "w");
        if (f != NULL)
        {
            fprintf(f, "time: %20.20e\n", t);
            fprintf(f, "stepid: %d\n", (int)step_id);
            fprintf(f, "nbReset: %d\n", (int)nbReset);
            fclose(f);
        }

        printf( "time: %20.20e\n", t);
        printf( "stepid: %d\n", (int)step_id);
        printf( "nbReset: %d\n", (int)nbReset);
    }

    //write numbered status (extra safety measure)
    {
        string numbered_status;
        char buf[500];
        sprintf(buf, "restart_%07d.status", (int)step_id);
        numbered_status = string(buf);

        FILE * f = fopen(numbered_status.c_str(), "w");
        if (f != NULL)
        {
            fprintf(f, "time: %20.20e\n", t);
            fprintf(f, "stepid: %d\n", (int)step_id);
            fprintf(f, "nbReset: %d\n", (int)nbReset);
            fclose(f);
        }
    }

    string numbered_filename;

    {
        char buf[500];
        sprintf(buf, "restart_%07d", (int)step_id);
        numbered_filename = string(buf);
    }

    //write grid
    IO_Binary<W,B> serializer;

     serializer.Write(*grid, "restart");
     serializer.Write(*grid, numbered_filename.c_str()); 

    printf("****SERIALIZING DONE****\n");

    {
        FILE * f = fopen("history.txt", step_id == 0? "w" : "a"); 
        if (f!= NULL)
        {
            fprintf(f, "%10.10f %d %d\n", t, (int)step_id, (int) nbReset);
            fclose(f);
        }
    }

    char buf[500];
    sprintf(buf, "restart.shape%04d", 0);
    string f(buf);

    myObstacle->save(t,f);
}

set<int> IF2D_FluidMediatedLau::_getBoundaryBlockIDs()
{
    set<int> result;
    vector<BlockInfo> vInfo = grid->getBlocksInfo();

    if (sRIGID_INLET_TYPE == "rigid_frame") //RIGID AT EACH BOUNDARY
        for(vector<BlockInfo>::iterator it = vInfo.begin(); it != vInfo.end(); it++)
        {
            const bool bX = it->index[0] == 0 || it->index[0] == pow(2.0, it->level)-1;
            const bool bY = it->index[1] == 0 || it->index[1] == pow(2.0, it->level)-1;

            if (bX || bY) result.insert(it->blockID);
        }
    else if (sRIGID_INLET_TYPE == "rigid_inlet_only") //RIGID ONLY AT THE INLET
        for(vector<BlockInfo>::iterator it = vInfo.begin(); it != vInfo.end(); it++)
        {
            const bool bX = it->index[0] == 0;

            if (bX) result.insert(it->blockID);
        }
    else if (sRIGID_INLET_TYPE == "free_outlet_only") //RIGID AT EACH BOUNDARY EXCEPT FOR THE OUTLET
        for(vector<BlockInfo>::iterator it = vInfo.begin(); it != vInfo.end(); it++)
        {
            const bool bX = it->index[0] == 0;
            const bool bY = it->index[1] == 0 || it->index[1] == pow(2.0, it->level)-1;

            if (bX || bY) result.insert(it->blockID);
        }
    else if (sRIGID_INLET_TYPE == "free_frame") //NON RIGID AT EACH BOUNDARY
        result.clear();
    else 
    {
        printf("IF2D_FluidMediatedLau::_getBoundaryBlockIDs: the chosen sRIGID_INLET_TYPE (=%s) is not supported. Aborting\n", sRIGID_INLET_TYPE.c_str());
        abort();
    }


    return result;
}

void IF2D_FluidMediatedLau::_dump(string filename)
{
    IF2D_Clear cleaner;
    cleaner.clearTmp(*grid);
    myObstacle->characteristic_function();

    IO_VTKNative<W,B, 4,0> vtkdumper;
    vtkdumper.Write(*grid, grid->getBoundaryInfo(), filename);
}

void IF2D_FluidMediatedLau::_dump()
{
    char buf[500];
    sprintf(buf, "avemaria_%07d", (int)step_id);
    _dump(buf);
}

void IF2D_FluidMediatedLau::_ic(Grid<W,B>& grid)
{
    vector<BlockInfo> vInfo = grid.getBlocksInfo();

    for(int i=0; i<(int)vInfo.size(); i++)
    {
        BlockInfo info = vInfo[i];
        B& b = grid.getBlockCollection()[vInfo[i].blockID];

        for(int iy=0; iy<B::sizeY; iy++)
            for(int ix=0; ix<B::sizeX; ix++)
            {
                b(ix, iy).omega = 0;
                b(ix, iy).u[0] = 0;
                b(ix, iy).u[1] = 0;
                b(ix, iy).tmp = 0;
            }
    }

    IF2D_Clear cleaner;
    cleaner.clearTmp(grid);
    myObstacle->characteristic_function();
}

void IF2D_FluidMediatedLau::_refine(bool bUseIC)
{
    if (bUNIFORM) return;

    set<int> boundary_blocks = _getBoundaryBlockIDs();

    // For initial condition refine given Xs only
    if (bUseIC)
    {
        while(true)
        {
            ((Refiner_BlackList*)refiner)->set_blacklist(&boundary_blocks);
            const int refinements = Science::AutomaticRefinement<0,0>(*grid, fwt_obstacle, RTOL, LMAX, 1, NULL, (void (*)(Grid<W,B>&))NULL, &boundary_blocks);
            _ic(*grid);
            if (refinements == 0) break;
        }
    }

    // For the rest refine given omega AND velocity
    if (!bREFINEOMEGAONLY)
    {
        while(true)
        {
            ((Refiner_BlackList*)refiner)->set_blacklist(&boundary_blocks);
            const int refinements = Science::AutomaticRefinement<0,2>(*grid, fwt_wuv, RTOL, LMAX, 1, NULL, (void (*)(Grid<W,B>&))NULL, &boundary_blocks);
            // printf("REFINE fwt_wuv \n");
            // const int refinements = Science::AutomaticRefinement<0,3>(*grid, fwt_wuvx, RTOL, LMAX, 1, NULL, (void (*)(Grid<W,B>&))NULL, &boundary_blocks);
            // printf("REFINE fwt_wuvx \n");
            // const int refinements = Science::AutomaticRefinement<0,0>(*grid, fwt_obstacle, RTOL, LMAX, 1, NULL, (void (*)(Grid<W,B>&))NULL, &boundary_blocks);
            // printf("REFINE fwt_obstacle \n");
            if (refinements == 0) break;
        }
    }
    else
    {
        while(true)
        {
            ((Refiner_BlackList*)refiner)->set_blacklist(&boundary_blocks);
            const int refinements = Science::AutomaticRefinement<0,0>(*grid, fwt_omega, RTOL, LMAX, 1, NULL, (void (*)(Grid<W,B>&))NULL, &boundary_blocks);
            if (refinements == 0) break;
        }
    }
}

void IF2D_FluidMediatedLau::_compress(bool bUseIC)
{
    if (bUNIFORM) return;

    set<int> boundary_blocks = _getBoundaryBlockIDs();

    // For initial condition compress given Xs only
    if (bUseIC)
    {
    IF2D_Clear cleaner;
    cleaner.clearTmp(*grid);
        myObstacle->characteristic_function();
        Science::AutomaticCompression<0,0>(*grid, fwt_obstacle, CTOL, 1, NULL, (void (*)(Grid<W,B>&))NULL);
        return;
    }

    // For the rest refine given omega AND velocity
    if (!bREFINEOMEGAONLY)
         Science::AutomaticCompression<0,2>(*grid, fwt_wuv, CTOL, 1, NULL, (void (*)(Grid<W,B>&))NULL);
        // Science::AutomaticCompression<0,3>(*grid, fwt_wuvx, CTOL, 1, NULL, (void (*)(Grid<W,B>&))NULL);
    else
        Science::AutomaticCompression<0,0>(*grid, fwt_omega, CTOL, 1, NULL, (void (*)(Grid<W,B>&))NULL);
}

double IF2D_FluidMediatedLau::_initial_dt(int nsteps) //take from SmartInteraction
{
    const Real dt_min = 1e-5;
    const Real dt_max = 1e-2;

    Real time = 0.0;
    for(unsigned int i=0; i<nsteps; i++)
        time += dt_min + (Real)(i)/(Real)(nsteps)*(dt_max - dt_min);

    Real time_current = 0.0;
    Real dt_current = 0.0;
    for(unsigned int i=0; i<nsteps; i++)
    {
        dt_current = dt_min + (Real)(i)/(Real)(nsteps)*(dt_max - dt_min);
        time_current += dt_current;
        if(time_current>t)
            break;
    }

    if( t<=time )
        return dt_current;
    else
        return 1000.0;

    return 1e-3;
}
// {
//  if ( step_id>=nsteps ) return 1000.0;
//
//  const double dt_min = 1e-7;//1e-6;
//  const double dt_max = 1e-4;//1e-3;
//  return dt_min + (double)(step_id)/(double)(nsteps)*(dt_max - dt_min);
//  //return 1e-3; //For convergence the best is ramp = 1 step with dt 1e-3
// }

void IF2D_FluidMediatedLau::_tnext(double &tnext, double &tnext_dump, double &tend)
{
    //0. setup
    //1. prepare candidate tnexts
    //2. pick the smallest one, report who wins

    //0.
    vector<double> tnext_candidates;
    vector<string> tnext_names;

    const double moduinf = sqrt(Uinf[0]*Uinf[0]+Uinf[1]*Uinf[1]);
    const double nondim_factor = (moduinf==0.0)?1.0:(moduinf*2.0/D);
    const double max_dx = (1./B::sizeX)*pow(0.5,grid->getCurrentMinLevel());
    const double min_dx = (1./B::sizeX)*pow(0.5,grid->getCurrentMaxLevel());
    const double max_vel = advection->compute_maxvel();

    //1.
    tend = TEND/nondim_factor;

    tnext_candidates.push_back(t + _initial_dt(RAMP));
    tnext_names.push_back("RAMP");

    tnext_candidates.push_back(t + max_dx/max_vel*CFL*0.999);
    cout << max_dx << " " <<  max_vel << " " << CFL <<  "CFL INFO"<< endl;
    tnext_names.push_back("CFL");

    if (LAMBDADT>0)
    {
        tnext_candidates.push_back(t + LAMBDADT/LAMBDA);
        tnext_names.push_back("LAMBDA");
    }

    if (bPARTICLES)
    {
        tnext_candidates.push_back(t + advection->estimate_largest_dt());
        tnext_names.push_back("LCFL");
    }

    if (bFMMSKIP)
    {
        tnext_candidates.push_back(t + 0.95*B::sizeX*min_dx/max_vel);
        tnext_names.push_back("CFL-FMMSKIP");
    }

    const bool bDiffusionLTS = true;
    if (bDiffusionLTS)
    {
        tnext_candidates.push_back(t + diffusion->estimate_largest_dt());
        tnext_names.push_back("LTS-Fc");
    }
    else                
    {
        tnext_candidates.push_back(t + diffusion->estimate_smallest_dt());
        tnext_names.push_back("GTS-Fc");
    }

    {
        const double delta = 1./(DUMPFREQ*nondim_factor);
        tnext_dump = (DUMPFREQ!=0)?(1.+floor(t/delta))*delta:10000.0;
    }

    //2.
    {
        tnext = tnext_candidates[0];
        int imin = 0;

        for(int i=1; i<(int)tnext_candidates.size(); i++)
            if (tnext_candidates[i] < tnext)
            {
                imin = i;
                tnext = tnext_candidates[i];
            }

        const string dtBound = tnext_names[imin] + " bound";

        printf("####################### t is %e, dt is %e, %s,\t{", t, tnext - t, dtBound.c_str());

        for(int i=0; i<(int)tnext_candidates.size(); i++)
            printf("%6s:%2.2e ", tnext_names[i].c_str(), tnext_candidates[i] - t);
        printf("}\n");

        FILE * f = fopen("report.txt", step_id == 0 ? "w" : "a");
        assert(f!=NULL);

        fprintf(f, "####################### t is %e, dt is %e and is bound by: %s", t, tnext - t, dtBound.c_str());
        fprintf(f, "stepid=%d\tT=%e\tDT=%e\t", (int)step_id, tnext*nondim_factor, (tnext - t)*nondim_factor);
        for(int i=0; i<(int)tnext_candidates.size(); i++)
            fprintf(f, "%s: %2.2e,\t", tnext_names[i].c_str(), tnext_candidates[i] - t);

        fprintf(f, "GTS-Fc: %2.2e,\t", FC*pow(min_dx,2)/(8.0*nu));

        fprintf(f, "\n");
        fclose(f);
    }
}



void IF2D_FluidMediatedLau::saveLearning()
{
    FILE* fid;
    fid = fopen("learning.txt", "a");
    fprintf(fid, "%d %10.6f %d %10.6f %10.6f\n", step_id, t, action, state, reward);
    fclose(fid);
}

void IF2D_FluidMediatedLau::_printMemoryConsumption()
{
  long peak_rss_bytes, current_rss_bytes;
  ComputationDiagnostics::print_memory_usage(peak_rss_bytes,current_rss_bytes);
    
  FILE * f = fopen("computation.txt", step_id == 1 && !bRESTART? "w" : "a");

  const double nofblocks = (double)grid->getBlocksInfo().size();
  const double nofgridpoints = grid->getBlocksInfo().size()*(double)(B::sizeX*B::sizeY*B::sizeZ);
    
  if (step_id == 1)
    {
      fprintf(f, "it\tt\tnofblocks\tnofgridpoints\tpeakMemMB\tcurrMemMB\n");
    }
    
  fprintf(f, "%d\t%e\t%e\t%e\t%e\t%e\n", step_id, t, nofblocks, nofgridpoints,peak_rss_bytes/(1024.*1024.),current_rss_bytes/(1024.*1024.));
  fclose(f);
}


void IF2D_FluidMediatedLau::act(const int action_)
{
  this->action = action_;
        myObstacle->act(action_);
}

double IF2D_FluidMediatedLau::getState()
{
        state = myObstacle->getState();
	return state;
}

double IF2D_FluidMediatedLau::getReward()
{
        reward = myObstacle->getReward();
	return reward;
}

double IF2D_FluidMediatedLau::getTime()
{
  return t;
}
void IF2D_FluidMediatedLau::run()
{
    const tbb::tick_count start_instant = tbb::tick_count::now();

    printf("\n\n\n\n------------------ STEP %d ------------------\n",(int)step_id);

    
    


    act(0); // Temporary (for test), swimmer take action to go straight





    if(step_id%ADAPTFREQ==0 and step_id>0) 
      {
        printf("REFINING..\n");
        profiler.push_start("REF");
        _refine(false); // don't use IC for refinement
        profiler.pop_stop();
        printf("DONE WITH REFINEMENT\n");
      }
    {
            printf("INIT STEP\n");

            // Create shape here and load div def into tmp
            profiler.push_start("SHAPE");
            IF2D_Clear cleaner;
            cleaner.clearTmp(*grid);
            myObstacle->create(t);

            profiler.pop_stop();
            printf("DONE WITH SHAPE CREATION\n");

            // Kill vorticity at right boundary
            if(bUSEKILLVORT)
            {
                killVort->killVorticity();
                printf("DONE WITH KILLING VORTICITY AT RIGHT BOUNDARY\n");
            }

            // Reconstruct velocity field from vorticity and potential
            profiler.push_start("VEL");
            velsolver->compute_velocity();
            printf("DONE WITH VELOCITY FROM VORTICITY\n");
            if(bUSEPOTENTIAL)
            {
                potsolver->compute_velocity();
                printf("DONE WITH VELOCITY FROM POTENTIAL\n");
            }
            profiler.pop_stop();

            double tnext, tnext_dump, tend;
            _tnext(tnext, tnext_dump, tend);
            const Real dt = (Real)tnext - t;
            printf("DONE WITH TNEXT\n");            

            profiler.push_start("DESVEL");

            myObstacle->computeDesiredVelocity(t);
            if(bADAPTVEL)                         
            {
                myObstacle->adaptVelocity(t,Uinf);
                advection->set_Uinfinity(Uinf);
                penalization->set_Uinfinity(Uinf); 
                // obstacle->set_Uinfinity dealt with in adaptVelocity
            }

            profiler.pop_stop();
            profiler.push_start("PEN");
            cleaner.clearTmp(*grid);
            myObstacle->characteristic_function();

            penalization->perform_timestep(dt);
            profiler.pop_stop();
            printf("DONE WITH PENALIZATION\n");

            myObstacle->computeDragAndStuff(t);
            printf("DONE WITH DIAGNOSTICS\n");

            profiler.push_start("DIFF");
            diffusion->perform_timestep(dt);
            profiler.pop_stop();
            printf("DONE WITH DIFFUSION\n");

            profiler.push_start("ADV");
            advection->perform_timestep(dt);
            profiler.pop_stop();
            printf("DONE WITH ADVECTION\n");

            char buf[500];
            sprintf(buf, "shape_%04d", 0);
            string f(buf);
            myObstacle->update(dt,t,f);
	    saveLearning(); 

                        t = tnext;
                        step_id++;

            // if(step_id%SAVEFREQ==0 || step_id <= 5)
            if(step_id%SAVEFREQ==0)
            {
                printf("DUMPING...\n");
                _dump(); 
                printf("DONE WITH DUMPING\n");
            }

#ifdef _FTLE_
            if(tnext >= tnext_dump)
            {
                printf("FTLE SAVING...\n");
                _save();
                printf("DONE FTLE SAVING\n");
            }
#endif

            if (t >= tend)
            {
                printf("T=TEND=%2.2f reached! (t=%2.2f)\n", TEND, t);
                exit(0);
            }

            {
                const tbb::tick_count now = tbb::tick_count::now();
                const Real nondim_factor = sqrt(Uinf[0]*Uinf[0]+Uinf[1]*Uinf[1])*2/D;
                const Real Tcurr = t*nondim_factor;
                const Real wallclock = (now-start_instant).seconds();
                const int nblocks = grid->getBlocksInfo().size();

                FILE * f = fopen("more-perfmon.txt", step_id == 1 ? "w" : "a");

                if (f!=NULL)
                {
                    if (step_id == 0)
                        fprintf(f,"stepid\twall-clock[s]\tT[-]\tblocks\tADV-rhs\tDIFF-rhs\n");

                    fprintf(f,"%d\t%e\t%e\t%d\t%d\t%d\n", (int)step_id, wallclock, Tcurr, nblocks, advection->get_nofrhs(), diffusion->get_nofrhs());

                    fclose(f);
                };

                _printMemoryConsumption();
            }

            printf("END TIME STEP %d\n", ADAPTFREQ);
        }

    if(step_id%ADAPTFREQ==0)
      {
        printf("COMPRESS..\n");
        profiler.push_start("COMPRESS");
        _compress(false);
        profiler.pop_stop();
        printf("DONE WITH COMPRESS\n");
      }
        profiler.printSummary();

#ifndef _FTLE_
        // Save should be after compressing otherwise one compressing stage would be lost!
        if(step_id % SAVEFREQ == 0)
        {
            printf("SAVING...\n");
            _save();
            printf("DONE SAVING\n");
        }
#endif
}
