/* Charles Pina - cpina@ucsc.edu
 * CMPS 161 Final Project
 * proj.c
 *
 * Description:
 *
 * The following program is a fluid dynamics simulation, meant to represent
 * the mixing of fluids (tea and milk) induced by stirring and poking.
 *
 * The mathematic solver is from Jos Stam's GDC2003 paper, "Real-Time Fluid
 * Dynamics for Games," and this program is a heavily modified version of his
 * 2d demonstration code.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <GLUT/glut.h>

/* macros */

#define IX(i,j) ((i)+(N+2)*(j))

#define PI 3.14159
#define RED_MIN 69.0f/255.0f
#define RED_MAX 1.0f
#define GREEN_MIN 36.0f/255.0f
#define GREEN_MAX 1.0f
#define BLUE_MIN 7.0f/255.0f
#define BLUE_MAX 1.0f

/* external definitions (from solver.c) */
extern void dens_step ( int N, float * x, float * x0, float * u, float * v, float diff, float dt );
extern void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt );

/* data structures */
struct coord {
    float x, y, z;
};

struct spoon_t {
    float s, t;
    float passive;
    float direction;
    int dip, stir;
    coord pos;
};

/* global variables */
static int N;
static float dt, diff, visc;
static float force, source;
static int dvel;
static spoon_t spoon;

static float * u, * v, * u_prev, * v_prev;
static float * dens, * dens_prev;
static float rx, ry, rz, zm;
double rot;

static int win_id;
static int win_x, win_y;
static int mouse_down[3];
static int omx, omy, mx, my;


/* 
 * free/clear/allocate simulation data
 */

/* free_data */
static void free_data ( void )
{
	if ( u ) free ( u );
	if ( v ) free ( v );
	if ( u_prev ) free ( u_prev );
	if ( v_prev ) free ( v_prev );
	if ( dens ) free ( dens );
	if ( dens_prev ) free ( dens_prev );
}

/* clear_data: clears and initializes the data */
static void clear_data ( void )
{
	int i, size=(N+2)*(N+2);

	for ( i=0 ; i<size ; i++ ) {
		u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;

        // initialize densities to be a gradient above 80%
        if(i>size*0.80) {
            float t = 5.0 * ((float)i/(float)size - 0.8);
            dens[i] = dens_prev[i] = t;
        }

        // and a gradient below 50%
        if(i<size*0.50) {
            float t = - 0.5 + (1.0*(float)i/(float)size);
            dens[i] = dens_prev[i] = t;
        }
	}
}

static int allocate_data ( void )
{
	int size = (N+2)*(N+2);

	u			= (float *) malloc ( size*sizeof(float) );
	v			= (float *) malloc ( size*sizeof(float) );
	u_prev		= (float *) malloc ( size*sizeof(float) );
	v_prev		= (float *) malloc ( size*sizeof(float) );
	dens		= (float *) malloc ( size*sizeof(float) );	
	dens_prev	= (float *) malloc ( size*sizeof(float) );

	if ( !u || !v || !u_prev || !v_prev || !dens || !dens_prev ) {
		fprintf ( stderr, "cannot allocate data\n" );
		return ( 0 );
	}

	return ( 1 );
}

/* map_red: returns the red component of my thai iced tea transfer function */
float map_red(float x) {
    return (1-x)*RED_MIN + (x)*RED_MAX;
}

/* map_green: returns the green component of the transfer function */
float map_green(float x) {
    return (1-x)*GREEN_MIN + (x)*GREEN_MAX;
}

/* map_blue: like red and green mapping functions but returns the blue
 * component */
float map_blue(float x) {
    return (1-x)*BLUE_MIN + (x)*BLUE_MAX;
}

/* circle_x: map t to parameterized circle's x, scaled by z */
float circle_x(float t, float z) {
    return (z+((1.0-z)/1.5))*sin(2*PI*t);
}

/* circle_x: map t to parameterized circle's y, scaled by z */
float circle_y(float t, float z) {
    return (z+((1.0-z)/1.5))*cos(2*PI*t);
}

/* calculate_position: return position given s (height), and t (position on
 * circle) of cylinder */
coord calculate_position(float s, float t) {
    coord pos;
    pos.x = circle_x(t, s);
    pos.y = circle_y(t, s);
    pos.z = s;
    return pos;
}

/* 
 * OpenGL specific drawing routines
 */

/* pre_display: sets up the screen, initial opengl options */
static void pre_display ( void )
{
    GLfloat light_ambient[] = {0.3f, 0.3f, 0.3f, 1.0f};
    GLfloat light_diffuse[] = {0.8f, 0.8f, 0.8f, 1.0f};
    GLfloat light_specular[] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat light_position[] = {1.0f, -1.0f, 1.0f, 0.0f};

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    glEnable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glEnable(GL_POLYGON_STIPPLE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glCullFace(GL_BACK);
    glShadeModel(GL_SMOOTH);
    glFrontFace(GL_CW);

    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity ();
	glViewport ( 0, 0, win_x, win_y );
    gluPerspective(60, win_x/win_y, 0.10f, 50.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glEnable(GL_COLOR_MATERIAL);
	glClearColor ( 0.3f, 0.3f, 0.3f, 1.0f );

    glEnable(GL_LINE_SMOOTH);

}

/* post_display */
static void post_display ( void )
{
	glutSwapBuffers ();
}

/* update_spoon: maps the cursor's cylindrical coordinates to xyz space */
static void update_spoon ( void ) {
    spoon.pos = calculate_position(spoon.s, spoon.t);
}

/* draw_velocity: draws the velocity field to the screen */
static void draw_velocity ( void ) {
	int i, j;
	float x, y, h;

	h = 1.0f/N;

    glPushMatrix();
    glScalef(2, 2, 2);
    glTranslatef(-0.5, -0.5, 0);
    glDisable(GL_LIGHTING);
	glColor3f ( 1.0f, 1.0f, 1.0f );
	glLineWidth ( 1.0f );
	glBegin ( GL_LINES );
    for ( i=1 ; i<=N ; i++ ) {
        x = (i-0.5f)*h;
        for ( j=1 ; j<=N ; j++ ) {
            y = (j-0.5f)*h;

            glVertex2f ( x, y );
            glVertex2f ( x+0.5*u[IX(i,j)], y+0.5*v[IX(i,j)] );
        }
    }
	glEnd ();
    glEnable(GL_LIGHTING);
    glPopMatrix();
}

static void draw_spoon ( void ) {
    coord p, c, n;
    update_spoon();
    p = calculate_position(1.0, spoon.t-0.02);
    c = calculate_position(0.75, spoon.t);
    n = calculate_position(1.0, spoon.t+0.02);

    glPushMatrix();
    glColor4f(1.0, 1.0, 1.0, 0.5);
    glScalef(0.6, 0.6, 1.0);
    glBegin(GL_POLYGON);
        glVertex3f(c.x, c.y, 0.9+(spoon.s/5)-0.05);
        glVertex3f(n.x, n.y, 0.9+(spoon.s/5)+0.05);
        glVertex3f(p.x, p.y, 0.9+(spoon.s/5)+0.05);
    glEnd();

    p = calculate_position(1.0, spoon.passive-0.02);
    c = calculate_position(0.75, spoon.passive);
    n = calculate_position(1.0, spoon.passive+0.02);

    glColor4f(1.0, 1.0, 1.0, 0.05);
    glBegin(GL_POLYGON);
        glVertex3f(c.x, c.y, 1.05);
        glVertex3f(n.x, n.y, 1.15);
        glVertex3f(p.x, p.y, 1.15);
    glEnd();

    glPopMatrix();
}

/* draw_cup: draws transparent cup, slightly larger than tea */
static void draw_cup ( void ) {
    int i, j;
    float x, y, h;

    h = 1.0f/N;

    glPushMatrix();
    glTranslatef(0, 0, -0.05);
    glScalef(0.55, 0.55, 1.10);
    glColor4f(1.0, 1.0, 1.0, 0.05);
    glMaterialf(GL_FRONT, GL_SHININESS, 100.0);
    for ( i=0; i<N; i++ ) {
        x = i*h;
        for (j=0; j<N; j++ ) {
            y = j*h;
            glBegin(GL_QUADS);
            glNormal3f ( -circle_y(x+h, y+h), circle_x(x+h, y+h), 0 );
            glVertex3f ( circle_x(x+h, y+h), 
                         circle_y(x+h, y+h), 
                         y+h );

            glNormal3f ( -circle_y(x, y+h), circle_x(x, y+h), 0 );
            glVertex3f ( circle_x(x, y+h), 
                         circle_y(x, y+h), 
                         y+h );

            glNormal3f ( -circle_y(x, y), circle_x(x, y), 0 );
            glVertex3f ( circle_x(x, y), 
                         circle_y(x, y), 
                         y );

            glNormal3f ( -circle_y(x+h, y), circle_x(x+h, y), 0 );
            glVertex3f ( circle_x(x+h, y), 
                         circle_y(x+h, y), 
                         y );
            glEnd();
        }
    }
    
    glPopMatrix();
}

/* draw_tea: draws a cylinder with the fluid mixture as its texture */
static void draw_tea ( void )
{
	int i, j;
	float x, y, h, d00, d01, d10, d11;

	h = 1.0f/N;

    glPushMatrix();
    glScalef(0.5, 0.5, 1);
    glMaterialf(GL_FRONT, GL_SHININESS, 0.0);
    for ( i=0 ; i<=N ; i++ ) {
        x = i*h;
        for ( j=0 ; j<=N ; j++ ) {
            y = j*h;

            d00 = dens[IX(i,j)];
            d01 = dens[IX(i,j+1)];
            d10 = dens[IX(i+1,j)];
            d11 = dens[IX(i+1,j+1)];

            glBegin ( GL_QUADS );
            glNormal3f ( -circle_y(x+h, y+h), circle_x(x+h, y+h), 0 );
            glColor3f  ( map_red(d11), map_green(d11), map_blue(d11) ); 
            glVertex3f ( circle_x(x+h, y+h), 
                         circle_y(x+h, y+h), 
                         y+h );

            glNormal3f ( -circle_y(x, y+h), circle_x(x, y+h), 0 );
            glColor3f  ( map_red(d01), map_green(d01), map_blue(d01) ); 
            glVertex3f ( circle_x(x, y+h), 
                         circle_y(x, y+h), 
                         y+h );

            glNormal3f ( -circle_y(x, y), circle_x(x, y), 0 );
            glColor3f  ( map_red(d00), map_green(d00), map_blue(d00) ); 
            glVertex3f ( circle_x(x, y), 
                         circle_y(x, y), 
                         y );

            glNormal3f ( -circle_y(x+h, y), circle_x(x+h, y), 0 );
            glColor3f  ( map_red(d10), map_green(d10), map_blue(d10) ); 
            glVertex3f ( circle_x(x+h, y), 
                         circle_y(x+h, y), 
                         y );
            glEnd ();
        }
    }

    glPopMatrix();
}

/* get_from_UI: relates mouse movements to forces sources */
static void get_from_UI ( float * d, float * u, float * v ) {
	int i, j, size = (N+2)*(N+2);

	for ( i=0 ; i<size ; i++ ) {
		u[i] = v[i] = d[i] = 0.0f;
	}

	if ( !mouse_down[0] && !mouse_down[2] ) return;

	i = (int)((       mx /(float)win_x)*N+1);
	j = (int)(((win_y-my)/(float)win_y)*N+1);

	if ( i<1 || i>N || j<1 || j>N ) return;

	if ( mouse_down[0] ) {
		u[IX(i,j)] = force * (mx-omx);
		v[IX(i,j)] = force * (omy-my);
	}

	if ( mouse_down[2] ) {
		d[IX(i,j)] = source;
	}

	omx = mx;
	omy = my;

	return;
}

/* key_func: callback for keyboard input */
static void key_func ( unsigned char key, int x, int y ) {
	switch ( key )
	{
		case 'c':
		case 'C':
			clear_data ();
			break;
		case 'q':
		case 'Q':
			free_data ();
			exit ( 0 );
			break;
		case 'v':
		case 'V':
			dvel = !dvel;
			break;
        case 'x':
            rx+=15;
            break;
        case 'y':
            ry+=15;
            break;
        case 'z':
            rz+=15;
            break;
	}
}

/* mouse_func: callback for mouseclicks. 
 * if left click, start dipping animation. 
 * if right click, start stirring animation. */
static void mouse_func ( int button, int state, int x, int y ) {
	omx = mx = x;
	omx = my = y;

    if(state==GLUT_DOWN) {
        switch (button) {
            case 0:
                spoon.dip = 1;
                break;
            case 2:
                spoon.t = (float)x/(float)win_x;
                spoon.stir = 1;
                break;
        }
    }
}

/* motion_func: calback for mouse movement (passive)
 * if stirring or dipping, don't update the main cursor.
 * always update the passive cursor. */
static void motion_func ( int x, int y )
{
    if(!spoon.stir && !spoon.dip) 
        spoon.t = (float)x/(float)win_x;
    spoon.passive = (float)x/(float)win_x;
}

/* reshape_func: callback for window reshape. */
static void reshape_func ( int width, int height )
{
	glutSetWindow ( win_id );
	glutReshapeWindow ( width, height );

	win_x = width;
	win_y = height;
}

/* idle_func: the animation function. called when not drawing the
 * scene. updates the spoon cursor position, recalculates vector field */
static void idle_func ( void )
{

    if(spoon.dip) {
        if(!spoon.direction) spoon.direction = -1;
        if(spoon.s < 0) 
            spoon.direction = 1;

        spoon.s += (float)spoon.direction*0.10;
        my = (1-spoon.s)*(float)win_y;

        mouse_down[0] = 1;
        
        if(spoon.s > 1) { 
           mouse_down[0] = 0;
           spoon.s = 1;
           spoon.t = spoon.passive;
           spoon.direction = -1;
           spoon.dip = 0;
        }
    } else if(spoon.stir) {
        if(!spoon.direction) spoon.direction = -1;
        if(spoon.s < 0)
            spoon.direction = 1;

        spoon.s += (float)spoon.direction*0.10;
        spoon.t -= 0.05;

        mx = spoon.t*(float)win_x;
        if(mx >= win_x) mx = mx-win_x;
        if(mx < 0) mx = mx+win_x;

        my = 0.75*(1-spoon.s)*(float)win_y + 0.25*(float)win_y;

        mouse_down[0] = 1;

        if(spoon.s > 1) {
            spoon.direction = -1;
            mouse_down[0] = 0;
            spoon.stir = 0;
            spoon.s = 1;
            spoon.t = spoon.passive;
        }
    }

	get_from_UI ( dens_prev, u_prev, v_prev );
	vel_step ( N, u, v, u_prev, v_prev, visc, dt );
	dens_step ( N, dens, dens_prev, u, v, diff, dt );

	glutSetWindow ( win_id );
	glutPostRedisplay ();
}

/* display_func: draws the scene */
static void display_func ( void )
{
	pre_display ();

	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    glLoadIdentity();
    glTranslatef(0, 0, -2.0f);

    if ( dvel ) {
        draw_velocity ();
    } else {
        glPushMatrix();
        glRotatef(rx, 1, 0, 0);
        glRotatef(ry, 0, 1, 0);
        glRotatef(rz, 0, 0, 1);
        glTranslatef(0, 0, -0.5);

        draw_tea ();
        draw_cup ();
        draw_spoon ();
        glPopMatrix();
    }

	post_display ();
}


/* open_glut_window: open a glut compatible window and set callbacks */
static void open_glut_window ( void )
{
	glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE | GLUT_MULTISAMPLE );

	glutInitWindowPosition ( 0, 0 );
	glutInitWindowSize ( win_x, win_y );
	win_id = glutCreateWindow ( "Charles Pina - Tasty Thai Iced Tea v.2" );

	glClearColor ( 0.3f, 0.3f, 0.3f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glutSwapBuffers ();
	glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glutSwapBuffers ();

	pre_display ();

	glutKeyboardFunc ( key_func );
	glutMouseFunc ( mouse_func );
	glutMotionFunc ( motion_func );
    glutPassiveMotionFunc ( motion_func );
	glutReshapeFunc ( reshape_func );
	glutIdleFunc ( idle_func );
	glutDisplayFunc ( display_func );
}


/* main */
int main ( int argc, char ** argv )
{
	glutInit ( &argc, argv );

    N = 128;
    dt = 0.05f;
    diff = 0.0f;
    visc = 0.0f;
    force = 10.0f;
    source = 100.0f;

	dvel = 0;

	if ( !allocate_data () ) exit ( 1 );
	clear_data ();

	win_x = 512;
	win_y = 512;

    /* orient camera, set up initial state */
    rot = 0; 
    zm=-2.0; 
    rx=-90.0; 
    ry=0.0; 
    rz=90.0;
    spoon.s = 1;
    spoon.t = 0;
    spoon.pos = calculate_position(spoon.s, spoon.t);

    /* create window */
	open_glut_window ();

    /* start GUI loop */
	glutMainLoop ();

	exit ( 0 );
}
