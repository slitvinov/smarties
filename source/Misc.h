/*
 *  Misc.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Settings.h"
#include "ErrorHandling.h"
#ifdef _RL_VIZ
#ifdef __APPLE__
#include "GLUT/glut.h"
#endif
#endif
#include "iostream"
using namespace ErrorHandling;

typedef unsigned char byte;

inline Real _angle(Real ux, Real uy, Real vx, Real vy)
{
	const Real anglev = atan2(uy, ux) / M_PI*180.0;
	const Real angled = atan2(vy, vx) / M_PI*180.0;
	Real angle = anglev-angled;
	if (angle >  180.0) angle -= 360;
	if (angle < -180.0) angle += 360;
	return angle;
}

inline Real _dist (Real x1, Real y1, Real x2, Real y2)
{
	return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
}

inline int _discretize(Real val, Real min, Real max, int levels, bool belowMin, bool aboveMax)
{
	int lvl = 0;
    //WTF!? int always truncates so you never return "levels", max is levels-1
	int totLvl = levels;
	
	if (belowMin) lvl++;
    if (aboveMax) levels--;
    
	if (max - min > 1e-6) lvl += (val - min)*levels / (max - min);
    else if ((val > min) && (val < max)) die("Bad interval of discretization\n");

	
	if (val < min) {
		if (belowMin) return 0;
		else return 0;
            //std::cout << "max=" << max << " min=" << min << " levels="<< levels << " val="<< val << endl;
	}
	
	if (val >= max) {
		if (aboveMax) return totLvl-1; // lvl = [0 to totLvl-1]
		else return totLvl-1;
            //std::cout << "max=" << max << " min=" << min << " levels="<< levels << " val="<< val << endl;
	}
	
	return lvl;
}

inline int _logDiscr(Real val, Real min, Real max, int levels, bool belowMin, bool aboveMax)
{
	if (max - min < 1e-6) die("Bad interval of discretization\n");
	int lvl = 0;
	int oldLevels = levels;
	
	if (belowMin) {
		levels--;
		lvl++;
	}
	if (aboveMax) levels--;
	
	Real x = (max - min) / (pow(2.0, levels) - 2);
    
	while (val - min > x) {
		x   = x*2;
		lvl++;
	}
	
	if (val < min) return 0;
	if (val > max) return oldLevels - 1;
	return lvl;
}

#ifdef _RL_VIZ

struct vec3
{
	Real x,y,z;
	vec3(Real x = 0, Real y = 0, Real z = 0):x(x), y(y), z(z) {};
};

inline vec3 cross(vec3 a, vec3 b)
{
	vec3 res;
	res.x = a.y*b.z - a.z*b.y;
	res.y = a.z*b.x - a.x*b.z;
	res.z = a.x*b.y - a.y*b.x;
	return res;
}

inline void _drawFullCircle(Real radius, Real xc, Real yc, Real r, Real g, Real b)
{
	const Real deg2rad = M_PI/180;

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
		Real degInRad = i*deg2rad;
		glNormal3d(0,0,1);
		glVertex2f(xc+cos(degInRad)*radius,yc+sin(degInRad)*radius);
	}
	glEnd();
	
	glPopMatrix();
}

inline void _drawSphere(Real radius, Real x, Real y, Real r, Real g, Real b)
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

inline void _drawArrow(Real size, Real x, Real y, Real vx, Real vy, Real IvI, Real r, Real g, Real b)
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
