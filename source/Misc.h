/*
 *  Misc.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#ifdef _RL_VIZ
#ifdef __APPLE__
#include "GLUT/glut.h"
#endif
#endif

#include "Settings.h"
#include "ErrorHandling.h"
using namespace ErrorHandling;

typedef unsigned char byte;

inline double _angle(double ux, double uy, double vx, double vy)
{
	const double anglev = atan2(uy, ux) / M_PI*180.0;
	const double angled = atan2(vy, vx) / M_PI*180.0;
	double angle = anglev-angled;
	if (angle >  180.0) angle -= 360;
	if (angle < -180.0) angle += 360;
	return angle;
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

struct vec3
{
	double x,y,z;
	vec3(double x = 0, double y = 0, double z = 0):x(x), y(y), z(z) {};
};

inline vec3 cross(vec3 a, vec3 b)
{
	vec3 res;
	res.x = a.y*b.z - a.z*b.y;
	res.y = a.z*b.x - a.x*b.z;
	res.z = a.x*b.y - a.y*b.x;
	return res;
}

inline void _drawFullCircle(double radius, double xc, double yc, double r, double g, double b)
{
	const double deg2rad = M_PI/180;

	float col[4]  = { (GLfloat)r, (GLfloat)g, (GLfloat)b, 1.0 };
	float colorSpec[4] = { 0.0, 0.0, 0.0, 1.0 };
	GLfloat shininess[] = {0};
	glMaterialfv(GL_FRONT, GL_DIFFUSE, col);
	glMaterialfv(GL_FRONT, GL_SPECULAR, colorSpec);
	glMaterialfv(GL_FRONT, GL_SHININESS, shininess);	
	
	glPushMatrix();
	
	glColor3f(r,g,b);
	glBegin(GL_POLYGON);
	for (int i=0; i<360; i++)
	{
		double degInRad = i*deg2rad;
		glNormal3d(0,0,1);
		glVertex2f(xc+cos(degInRad)*radius,yc+sin(degInRad)*radius);
	}
	glEnd();
	
	glPopMatrix();
}

inline void _drawSphere(double radius, double x, double y, double r, double g, double b)
{
	float col[4]  = { (GLfloat)r, (GLfloat)g, (GLfloat)b, 1.0 };
	float colorSpec[4] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat shininess[] = {500};
	glMaterialfv(GL_FRONT, GL_DIFFUSE, col);
	glMaterialfv(GL_FRONT, GL_SPECULAR, colorSpec);
	glMaterialfv(GL_FRONT, GL_SHININESS, shininess);
	
	glPushMatrix();
	glColor3f(r,g,b);
	glTranslated(x,y,0);
	glutSolidSphere(radius, 16,16);
	glPopMatrix();
}

inline void _drawArrow(double size, double x, double y, double vx, double vy, double IvI, double r, double g, double b)
{
	float col[4]  = { (GLfloat)r, (GLfloat)g, (GLfloat)b, 1.0 };
	float colorSpec[4] = { 1.0, 1.0, 1.0, 1.0 };
	GLfloat shininess[] = {150};
	glMaterialfv(GL_FRONT, GL_DIFFUSE, col);
	glMaterialfv(GL_FRONT, GL_SPECULAR, colorSpec);
	glMaterialfv(GL_FRONT, GL_SHININESS, shininess);
	
	glPushMatrix();
	glColor3f(r,g,b);
	glTranslated(x,y,0);
	
	//glutSolidSphere(size, 16,16);
	glScaled(size/IvI, size/IvI, size/IvI);
	glBegin(GL_TRIANGLES);
	{
		vec3 a1(vx + vy/3, vy - vx/3, 0);
		vec3 b1(3*vx/4, 3*vy/4, -IvI);
		vec3 n1 = cross(a1, b1);
		
		glNormal3d(n1.x, n1.y, n1.z);
		glVertex3d(vx/2, vy/2, 0);
		glNormal3d(n1.x, n1.y, n1.z);
		glVertex3d(-vx/2 - vy/3, -vy/2 + vx/3, 0);
		glNormal3d(n1.x, n1.y, n1.z);
		glVertex3d(-vx/4, -vy/4, IvI);
		
		vec3 a2(3*vx/4, 3*vy/4, -IvI);
		vec3 b2(vx - vy/3, vy + vx/3, 0);
		vec3 n2 = cross(a2, b2);
		
		glNormal3d(n2.x, n2.y, n2.z);
		glVertex3d(vx/2, vy/2, 0);
		glNormal3d(n2.x, n2.y, n2.z);
		glVertex3d(-vx/4, -vy/4, IvI);
		glNormal3d(n2.x, n2.y, n2.z);
		glVertex3d(-vx/2 + vy/3, -vy/2 - vx/3, 0);
	}
	glEnd();
	
	glPopMatrix();
	glDisable(GL_COLOR_MATERIAL);
}
#endif
