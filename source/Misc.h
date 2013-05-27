/*
 *  Misc.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "ErrorHandling.h"
using namespace ErrorHandling;

inline double _angle(double ux, double uy, double vx, double vy)
{
	const double anglev = atan2(uy, ux) / M_PI*180.0;
	const double angled = atan2(vy, vx) / M_PI*180.0;
	const double angle = anglev-angled;
	return (angle<0.0)?angle+360.0:angle;
}

inline double _dist (double x1, double y1, double x2, double y2)
{
	return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
}

inline int _discretize(double val, double min, double max, int levels, bool belowMin, bool aboveMax)
{
	if (max - min < 1e-6) die("Bad interval of discretization\n");
	int lvl = 0;
	int oldLevels = levels;
	
	if (belowMin)
	{
		levels--;
		lvl++;
	}
	if (aboveMax) levels--;
	
	lvl += (val - min)*levels / (max - min);
	
	if (val < min)
	{
		if (belowMin) return 0;
		else die("Discretized value is below minimum\n");
	}
	
	if (val > max)
	{
		if (aboveMax) return oldLevels - 1;
		else die("Discretized value is above maximum\n");
	}
	
	return lvl;
}

inline int _logDiscr(double val, double min, double max, int levels, bool belowMin, bool aboveMax)
{
	if (max - min < 1e-6) die("Bad interval of discretization\n");
	int lvl = 0;
	int oldLevels = levels;
	
	if (belowMin)
	{
		levels--;
		lvl++;
	}
	if (aboveMax) levels--;
	
	double x = (max - min) / (pow(2.0, levels) - 2);	
	while (val - min > x)
	{
		x   = x*2;
		lvl++;
	}
	
	if (val < min)
	{
		if (belowMin) return 0;
		else die("Discretized value is below minimum\n");
	}
	
	if (val > max)
	{
		if (aboveMax) return oldLevels - 1;
		else die("Discretized value is above maximum\n");
	}
	
	return lvl;
}

#ifdef _RL_VIZ
inline void _drawFullCircle(double radius, double xc, double yc, double r, double g, double b)
{
	const double deg2rad = M_PI/180;
	
	glPushMatrix();
	
	glColor3f(r,g,b);
	glBegin(GL_POLYGON);
	for (int i=0; i<360; i++)
	{
		double degInRad = i*deg2rad;
		glVertex2f(xc+cos(degInRad)*radius,yc+sin(degInRad)*radius);
	}
	glEnd();
	
	glPopMatrix();
}

inline void _drawSphere(double radius, double x, double y, double r, double g, double b)
{
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	GLfloat lightpos[] = {0, 0, -1.0, 0};
	glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
	GLfloat lightColor[] = {r,g,b,1};
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor);
	
	glPushMatrix();
	glColor3f(r,g,b);
	glTranslated(x,y,0);
	glutSolidSphere(radius, 16,16);
	glPopMatrix();
}

inline void _drawArrow(double size, double x, double y, double vx, double vy, double IvI, double r, double g, double b)
{
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	GLfloat lightpos[] = {0, 0, -1.0, 0};
	glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
	GLfloat lightColor[] = {r,g,b,1};
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor);
	
	glPushMatrix();
	glColor3f(r,g,b);
	glTranslated(x,y,0);
	glScaled(size/IvI, size/IvI, 1);
	glBegin(GL_TRIANGLES);
	{
		glVertex3d(vx/2, vy/2, 0);
		glVertex3d(-vx/2 - vy/3, -vy/2 + vx/3, 0);
		glVertex3d(-vx/4, -vy/4, IvI);
		
		glVertex3d(vx/2, vy/2, 0);
		glVertex3d(-vx/4, -vy/4, IvI);
		glVertex3d(-vx/2 + vy/3, -vy/2 - vx/3, 0);
	}
	glEnd();
	
	glPopMatrix();	
}
#endif
