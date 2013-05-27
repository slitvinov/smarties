/*
 *  CellList.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <list>
#include <map>

using namespace std;

template <typename Object>
class Cells
{
public:
	int nx, ny;
	int nTot;
	
	double x0, x1, y0, y1;
	double hx, hy;
	double lx, ly;
	
	list<int>*        myObjects;
	vector<Object*>   objects;
	map<Object*, int> objMap;
	
public:
	
	inline int  getCellIndByIJ(int, int);
	inline void getCellIJByInd(int, int&, int&);
	inline int  which(int);
	inline int  getObjId(Object*);
	inline bool ifBelong(int, int);
	inline list<int>::iterator getPartListStart(int);
	inline list<int>::iterator getPartListEnd(int);
	
	Cells(vector<Object*>, double, double, double, double, double);
	void migrate();
};

template <typename Object>
Cells<Object>::Cells(vector<Object*> obj, double size, double xLeft, double yBottom, double xRight, double yTop):
objects(obj), x0(xLeft), x1(xRight), y0(yBottom), y1(yTop)
{
	lx = x1 - x0;
	ly = y1 - y0;
	
	nx = lx / size;
	ny = ly / size;
	if (nx == 0) nx = 1;
	if (ny == 0) ny = 1;
	hx = lx / nx;
	hy = ly / ny;
	
	nTot = nx * ny;
	
	myObjects = new list<int> [nTot];
	
	for (int pid = 0; pid < objects.size(); pid++)
	{
		objMap[objects[pid]] = pid;
		myObjects[which(pid)].push_back(pid);
	}
}

template <typename Object>
inline int Cells<Object>::getCellIndByIJ(int i, int j)
{
	return i*ny + j;	
}

template <typename Object>
inline void Cells<Object>::getCellIJByInd(int id, int& i, int& j)
{
	j = id % ny;
	i = id / ny;
}

template <typename Object>
inline int Cells<Object>::which(int pid)
{
	int i = floor((objects[pid]->x - x0) / hx);
	int j = floor((objects[pid]->y - y0) / hy);
	if (i >= nx) i = nx-1;
	if (i <  0)  i = 0;
	if (j >= ny) j = ny-1;
	if (j <  0)  j = 0;
	
	return getCellIndByIJ(i, j);
}

template <typename Object>
inline int Cells<Object>::getObjId(Object* obj)
{
	return objMap[obj];
}


template <typename Object>
inline bool Cells<Object>::ifBelong(int pid, int cid)
{
	return cid == which(pid);
}

template <typename Object>
inline list<int>::iterator Cells<Object>::getPartListStart(int cid)
{
	return myObjects[cid].begin();
}

template <typename Object>
inline list<int>::iterator Cells<Object>::getPartListEnd(int cid)
{
	return myObjects[cid].end();
}

template <typename Object>
void Cells<Object>::migrate()
{
	for (int cid=0; cid < nTot; cid++)
	{
		list<int>::iterator pidIter;
		pidIter = myObjects[cid].begin();
		
		while (pidIter != myObjects[cid].end())
		{
			int pid = *pidIter;
			if (!ifBelong(pid, cid))
			{
				myObjects[which(pid)].push_back(pid);
				pidIter = myObjects[cid].erase(pidIter);
			}
			else
			{
				pidIter++;
			}
		}
	}
}


//**********************************************************************************************************************
// CellsTraverser
// Instance of this class is private on all threads
//**********************************************************************************************************************
enum Direction
{
	None = 0, 
	East, 
	NorthEast, 
	North, 
	NorthWest, 
	West, 
	SouthWest, 
	South, 
	SouthEast,
	Incorrect
};

template <typename Object>
class CellsTraverser
{
private:
	Cells<Object>* cells;
	
	Direction state;
	int curId;
	list<int>::const_iterator curPart;
	double xAdd, yAdd;
	int origId;
	
	inline void moveToNext();
	
public:
	CellsTraverser(Cells<Object>* c):cells(c) {};
	void prepare(int);
	bool getNextXY(double&, double&, Object*&);
};


template <typename Object>
inline void CellsTraverser<Object>::moveToNext()
{
	// +----+----+----+
	// | NW | N  | NE |
	// +----+----+----+
	// | W  | 0  | E  |
	// +----+----+----+
	// | SW | S  | SE |
	// +----+----+----+
	
	int i, j;
	cells->getCellIJByInd(curId, i, j);
	
	switch (state)
	{
		case None:
			i++;
			state = East;
			break;
			
		case East:
			j++;
			state = NorthEast;
			break;
			
		case NorthEast:
			i--;
			state = North;
			break;
			
		case North:
			i--;
			state = NorthWest;
			break;
			
		case NorthWest:
			j--;
			state = West;
			break;
			
		case West:
			j--;
			state = SouthWest;
			break;
			
		case SouthWest:
			i++;
			state = South;
			break;
			
		case South:
			i++;
			state = SouthEast;
			break;
			
		case SouthEast:
			state = Incorrect;
			break;
			
		default:
			state = Incorrect;
	}
	
	if (i < 0)
	{
		xAdd -= cells->lx;
		i = cells->nx-1;
	}
	if (i >= cells->nx)
	{
		xAdd += cells->lx;
		i = 0;
	}
	if (j < 0)
	{
		yAdd -= cells->ly;
		j = cells->ny-1;
	}
	if (j >= cells->ny)
	{
		yAdd += cells->ly;
		j = 0;
	}
	
	curId = cells->getCellIndByIJ(i, j);
}

template <typename Object>
void CellsTraverser<Object>::prepare(int id)
{
	state = None;
	curId = cells->which(id);
	curPart = cells->getPartListStart(curId);
	xAdd = yAdd = 0;
	origId = id;
}

template <typename Object>
bool CellsTraverser<Object>::getNextXY(double& x, double& y, Object*& obj)
{
	while (state != Incorrect && 
		   curPart == cells->getPartListEnd(curId))	//  OK, this cell is finished
	{												//  Loop is here only to skip all empty cells
		moveToNext();								//  Choose next adjacent cell
													// and set shifts for x's and y's if we jump over a boundary
		curPart = cells->getPartListStart(curId);	//  Start to iterate over the list of current cell
	}
	
	if (state == Incorrect) return false;			//  Now we've done with traversing all the adjacent cells
	
	int id = *curPart;
	x   = xAdd + cells->objects[id]->x;				//  If we're here we have to return
	y   = yAdd + cells->objects[id]->y;				// the corrected coordinates of next object,
	obj = cells->objects[id];						// the object itself
	curPart++;										// and go to the next one. Voila.
	
	// Ooops. What if we try to return the original particle?
	// Just go to the next one in that case
	if (state == None && id == origId) return getNextXY(x, y, obj);
	return true;
}
