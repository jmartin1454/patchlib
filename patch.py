# Fri May 24 10:11:45 CDT 2019 Jeff added this line.
# June 16, 2019 Jeff modified to limit to class definitions.
#July 19,2024 Modeste added the draw option to output the layout of the MSR walls starting at line 253.

from scipy.constants import mu_0, pi
import numpy as np
from .Arrow3D import *

np.random.seed(45634634)

def b_segment(i,p0,p1,r):
    # p0 is one end (vector in m)
    # p1 is the other (m)
    # r is the position of interest (m)
    # i is the current (A) (flowing from p0 to p1, I think)
    d0 = r - p0
    d1 = r - p1
    ell = p1 - p0
    lend0 = np.sqrt(d0.dot(d0))
    lend1 = np.sqrt(d1.dot(d1))
    lenell = np.sqrt(ell.dot(ell))
    b_total = np.array([0,0,0])
    if(lenell!=0): # watch out for repeated points
        costheta0 = np.inner(ell,d0)/lenell/lend0
        costheta1 = -np.inner(ell,d1)/lenell/lend1
        ellcrossd0 = np.cross(ell,d0)
        lenellcrossd0 = np.sqrt(ellcrossd0.dot(ellcrossd0))
        modsintheta0 = lenellcrossd0/lenell/lend0
        a = lend0 * modsintheta0
        if(lenellcrossd0>0):
            nhat = ellcrossd0/lenellcrossd0
        else:
            nhat = np.array([0,0,0])

        if(a>0):
            b_total=mu_0*i/4.0/pi/a*(costheta0+costheta1)*nhat
        else:
            b_total = np.array([0,0,0])

    return b_total

def b_segment_2(i,p0,p1,x,y,z):
    # p0 is one end (vector in m)
    # p1 is the other (m)
    # x,y,z is the position of interest (m)
    # i is the current (A) (flowing from p0 to p1, I think)
    d0x=x-p0[0]
    d0y=y-p0[1]
    d0z=z-p0[2]
    d1x=x-p1[0]
    d1y=y-p1[1]
    d1z=z-p1[2]
    ell=p1-p0
    lend0=np.sqrt(d0x**2+d0y**2+d0z**2)
    lend1=np.sqrt(d1x**2+d1y**2+d1z**2)
    lenell=np.sqrt(ell.dot(ell))
    b_total_x=0.*x
    b_total_y=0.*y
    b_total_z=0.*z
    if(lenell!=0): # watch out for repeated points
        costheta0=(ell[0]*d0x+ell[1]*d0y+ell[2]*d0z)/lenell/lend0
        costheta1=-(ell[0]*d1x+ell[1]*d1y+ell[2]*d1z)/lenell/lend1
        ellcrossd0x=ell[1]*d0z-ell[2]*d0y
        ellcrossd0y=ell[2]*d0x-ell[0]*d0z
        ellcrossd0z=ell[0]*d0y-ell[1]*d0x
        lenellcrossd0=np.sqrt(ellcrossd0x**2+ellcrossd0y**2+ellcrossd0z**2)
        modsintheta0=lenellcrossd0/lenell/lend0
        a=lend0*modsintheta0
        nhatx=np.divide(ellcrossd0x,lenellcrossd0,out=np.zeros_like(ellcrossd0x*lenellcrossd0),where=lenellcrossd0!=0)
        nhaty=np.divide(ellcrossd0y,lenellcrossd0,out=np.zeros_like(ellcrossd0y*lenellcrossd0),where=lenellcrossd0!=0)
        nhatz=np.divide(ellcrossd0z,lenellcrossd0,out=np.zeros_like(ellcrossd0z*lenellcrossd0),where=lenellcrossd0!=0)

        pre=mu_0*i/4.0/pi*(costheta0+costheta1)
        b_total_x=np.divide(pre*nhatx,a,out=np.zeros_like(pre*nhatx),where=a!=0)
        b_total_y=np.divide(pre*nhaty,a,out=np.zeros_like(pre*nhaty),where=a!=0)
        b_total_z=np.divide(pre*nhatz,a,out=np.zeros_like(pre*nhatz),where=a!=0)

    return b_total_x,b_total_y,b_total_z

def b_loop(i,points,r):
    # i is the current (A)
    # points is a list of numpy 3-arrays defining the loop (m)
    # r is the position of interest (m)
    # returns the magnetic field as a numpy 3-array (T)
    # Note:  assumes that points[-1] to points[0] should be counted (closed loop)
    # This is why j starts at zero in the loop below
    b_total = np.array([0,0,0])
    for j in range(len(points)):
        b_total = b_total + b_segment(i,points[j-1],points[j],r)
    return b_total

def b_loop_2(i,points,x,y,z):
    # i is the current (A)
    # points is a list of numpy 3-arrays defining the loop (m)
    # x,y,z is the position of interest (m)
    # returns the magnetic field components (T)
    # Note:  assumes that points[-1] to points[0] should be counted (closed loop)
    # This is why j starts at zero in the loop below
    b_total_x=0.*x
    b_total_y=0.*y
    b_total_z=0.*z
    for j in range(len(points)):
        b_seg_x,b_seg_y,b_seg_z=b_segment_2(i,points[j-1],points[j],x,y,z)
        b_total_x=b_total_x+b_seg_x
        b_total_y=b_total_y+b_seg_y
        b_total_z=b_total_z+b_seg_z
    return b_total_x,b_total_y,b_total_z

class coil:
    def __init__(self,points,current):
        self.points=points
        self.current=current
    def set_current(self,current):
        self.current=current
    def b(self,r):
        return b_loop(self.current,self.points,r)
    def b_prime(self,x,y,z):
        return b_loop_2(self.current,self.points,x,y,z)
    def flip_this_coil(self):
        self.points=np.flip(self.points,axis=0)
    def move(self,dx,dy,dz):
        self.points[:,0]=self.points[:,0]+dx
        self.points[:,1]=self.points[:,1]+dy
        self.points[:,2]=self.points[:,2]+dz
    def wiggle(self,sigma): # wiggles each point according to normal distribution
        self.points=self.points+np.random.normal(0,sigma,(len(self.points),3))
    
class coilset:
    def __init__(self):
        self.coil=[]
        self.numcoils=len(self.coil)

    def add_coil(self,points):
        c=coil(points,0.0)
        self.coil.append(c)
        self.numcoils=len(self.coil)
        
    def swap_coils(self,n1,n2):   #numbering the coils
        self.coil[n1],self.coil[n2]=self.coil[n2],self.coil[n1]
        
    def set_current_in_coil(self,coilnum,i):
        if(coilnum<self.numcoils):
            self.coil[coilnum].set_current(i)
        else:
            print("Error %d is larger than number of coils %d"%coilnum,self.numcoils)
            
    def set_common_current(self,i):
        for coilnum in range(self.numcoils):
            self.set_current_in_coil(coilnum,i)

    def wiggle(self,sigma):
        for coil in self.coil:
            coil.wiggle(sigma)
            #coil.wiggle_up(sigma)

    def move(self,x,y,z):
        for coil in self.coils:
            coil.move(x,y,z)
            
    def zero_currents(self):
        for coilnum in range(self.numcoils):
            self.set_current_in_coil(coilnum,0.0)
                    
    def set_currents(self,veci):
        for coilnum in range(self.numcoils):
            self.set_current_in_coil(coilnum,veci[coilnum])

    def b(self,r):
        b_total=0.
        for coilnum in range(self.numcoils):
            b_total=b_total+self.coil[coilnum].b(r)
        return b_total
        
    def b_prime(self,x,y,z):
        b_total_x=0.*x
        b_total_y=0.*y
        b_total_z=0.*z
        for coilnum in range(self.numcoils):
            b_coil_x,b_coil_y,b_coil_z=self.coil[coilnum].b_prime(x,y,z)
            b_total_x=b_total_x+b_coil_x
            b_total_y=b_total_y+b_coil_y
            b_total_z=b_total_z+b_coil_z
        return b_total_x,b_total_y,b_total_z
        
    def draw_coil(self,number,ax,style,color):
        coil = self.coil[number]
        points = coil.points
        points=np.append(points,[points[0]],axis=0) # force draw closed loop
        x = ([p[0] for p in points])
        y = ([p[1] for p in points])
        z = ([p[2] for p in points])
        ax.plot(x,y,z,style,color=color)
        a=Arrow3D([x[0],x[1]],[y[0],y[1]],[z[0],z[1]],mutation_scale=20, 
                  lw=3,arrowstyle="-|>",color="r")
        ax.add_artist(a)
        ax.text(np.average(x),np.average(y),np.average(z),"%d"%number,color="r",
                horizontalalignment='center',
                verticalalignment='center')
        # ax.text(x[0],y[0],z[0],"%d"%number,color="r")

    def draw_coils(self,ax,style='-',color='black'):
        for number in range(self.numcoils):
            self.draw_coil(number,ax,style,color)

    def output_solidworks(self,outfile):
        with open(outfile,'w') as f:
            for number in range(self.numcoils):
                coil = self.coil[number]
                points = coil.points
                firstpoint=points[0]
                lastpoint=points[-1]
                if (not(firstpoint[0]==lastpoint[0] and
                        firstpoint[1]==lastpoint[1] and
                        firstpoint[2]==lastpoint[2])):
                    points=np.append(points,[points[0]],axis=0) # force draw closed loop
                for p in points:
                    f.write("{0}\t{1}\t{2}\n".format(p[0],p[1],p[2]))

    def output_scad(self,outfile,thickness=0.001):
        with open(outfile,'w') as f:
            f.write("module line(start, end, thickness = %f) {\n"%thickness)
            f.write("hull() {\n")
            f.write("translate(start) sphere(thickness);\n")
            f.write("translate(end) sphere(thickness);\n")
            f.write("}\n")
            f.write("}\n")
            for number in range(self.numcoils):
                coil = self.coil[number]
                points = coil.points
                firstpoint=points[0]
                lastpoint=points[-1]
                if (not(firstpoint[0]==lastpoint[0] and
                        firstpoint[1]==lastpoint[1] and
                        firstpoint[2]==lastpoint[2])):
                    points=np.append(points,[points[0]],axis=0) # force draw closed loop
                f.write("// coil %d\n"%number)
                for i in range(len(points)):
                    lastpoint=points[i-1]
                    thispoint=points[i]
                    f.write("line([%f,%f,%f],[%f,%f,%f]);\n"%(lastpoint[0],lastpoint[1],lastpoint[2],thispoint[0],thispoint[1],thispoint[2]))

    def output_scad_prime(self,outfile,thickness=0.001):
        with open(outfile,'w') as f:
            f.write("thickness = %f\n"%thickness)
            f.write("translate(end) sphere(thickness);\n")
            for number in range(self.numcoils):
                coil = self.coil[number]
                points = coil.points
                firstpoint=points[0]
                lastpoint=points[-1]
                if (not(firstpoint[0]==lastpoint[0] and
                        firstpoint[1]==lastpoint[1] and
                        firstpoint[2]==lastpoint[2])):
                    points=np.append(points,[points[0]],axis=0) # force draw closed loop
                f.write("// coil %d\n"%number)
                f.write("hull() {\n")
                for p in points:
                    f.write("translate([%f,%f,%f]) sphere(thickness);\n"%(p[0],p[1],p[2]))
                f.write("}\n")
 #Drawing layout of coils added by Modeste from Mark  2024/07/19
    def make_closed(self):
        '''
        iterate over all coils in the coil_set and make the coils closed if the first and last point in the list are not identical.
        
        returns number of coils closed
        '''
        count = 0
        for coil in self.coil:
            if np.any(coil.points[0] != coil.points[-1]):
                coil.points = np.vstack((coil.points,coil.points[0]))
                count = count + 1
        return count
        
    def make_open(self):
        '''
        iterate over all coils in the coil_set and make the coils open if the first and last point in the list are identical.
        
        returns number of coils opened.
        '''
        count = 0
        for coil in self.coil:
            if np.all(coil.points[0] == coil.points[-1]):
                coil.points = coil.points[:-1]
                count = count + 1
        return count
    def draw_pts(self, ax, points, sorting, equal=True, arrow=True, poslabel=False,color=None, **plt_kwargs):
        '''
        ax - matplotlib axis to draw on
        points- n x 3 array of points to be drawn
        sorting = 1x3 array indicating columns for [x-axis in plot, y-axis in plot, filter axis for pmin and pmax]
        equal - bool if axis should be axis scaling
        arrow - bool if arrows showing line directions should be drawn
        poslabel - if each vertex should be labels with its position, can be really slow if a large number of points (>50) are used
        '''

        p=ax.plot(points[:,sorting[0]],points[:,sorting[1]],**plt_kwargs)[0]
        if equal: ax.axis('equal')
        if poslabel:
            for lx,ly,x,y,z in zip(points[:,sorting[0]], points[:,sorting[1]],points[:,0],points[:,1],points[:,2]):
                ax.annotate(
                    '(%.1f,%.1f,%0.1f)'%(x,y,z),
                    (lx, ly),
                    fontsize=14,
                    color=p.get_color())
        if arrow:
          for i in range(len(points[:,sorting[0]])-1):
              ax.arrow(
                  points[i,sorting[0]], points[i,sorting[1]],
                  (points[i+1,sorting[0]]-points[i,sorting[0]])/3,
                  (points[i+1,sorting[1]]-points[i,sorting[1]])/3,
                  color=p.get_color(),
                  width=0.02,
                  length_includes_head=True
              )
    def draw_xy(self,ax,filter_axis=-1,equal=True,arrow=False,poslabel=False,**plt_kwargs):
        for coil in self.coil:
            self.draw_pts(ax,coil.points,[0,1,filter_axis],equal,arrow,poslabel,**plt_kwargs)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    def draw_yz(self,ax,filter_axis=-1,equal=True,arrow=True,poslabel=False,**plt_kwargs):
        for coil in self.coil:
            self.draw_pts(ax,coil.points,[1,2,filter_axis],equal,arrow,poslabel,**plt_kwargs)
        ax.set_xlabel('y (m)')
        ax.set_ylabel('z (m)')
    def draw_xz(self,ax,filter_axis=-1,equal=True,arrow=True,poslabel=False,**plt_kwargs):
        for coil in self.coil:
            self.draw_pts(ax,coil.points,[0,2,filter_axis],equal,arrow,poslabel,**plt_kwargs)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('z (m)')
    def draw_iso(self,ax,filter_axis=-1,equal=True,arrow=False,poslabel=False,**plt_kwargs):
        for coil in self.coil:
            ax.plot(coil.points[:,0],coil.points[:,1],coil.points[:,2],**plt_kwargs)
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_zlabel('z (m)')
        if equal:
            ax.set_xlim3d(-1.1, 1.1)
            ax.set_ylim3d(-1.1, 1.1)
            ax.set_zlim3d(-1.1, 1.1)
        if arrow:
            print("Cannot add arrows to 3d plot")
        if poslabel:
            print("cannot add position labels to 3d plot")
    def draw_layout(self,fig, equal= True,arrow=False,poslabel=False, title_add = "",drawL5=False,drawL6=False, **plt_kwargs):
        
        self.make_closed()
        
        if not fig.axes:
            ax3=[]
            ax3.append(fig.add_subplot(2, 2, 3)) #lower left, xy-plane
            ax3.append(fig.add_subplot(2, 2, 4)) #lower right,yz-plane
            ax3.append(fig.add_subplot(2, 2, 1)) #upper left, xz-plane
            ax3.append(fig.add_subplot(2, 2, 2, projection='3d')) #upper right isometric view
        else:
            ax3 = fig.axes
        
        ax3[0].set_title("North and South Walls")
        self.draw_xz(ax3[0],equal=equal,arrow=arrow,poslabel=poslabel,**plt_kwargs)
        
        ax3[1].set_title("East and West Walls")
        self.draw_yz(ax3[1],equal=equal,arrow=arrow,poslabel=poslabel,**plt_kwargs)
        
        ax3[2].set_title("Floor and Ceiling")
        self.draw_xy(ax3[2],equal=equal,arrow=arrow,poslabel=poslabel,**plt_kwargs)
        
        self.draw_iso(ax3[3])
        
        fig.suptitle("Orthgraphic View of Traces"+title_add)
        
        if drawL5:
            DrawLayer5(ax3[0],ax3[1],ax3[2])
        if drawL6:
            DrawLayer6(ax3[0],ax3[1],ax3[2])
        
        self.make_open()
        
        return ax3
    
def DrawLayer5(axz,ayz,axy, lcolor = "magenta", width = 1):
    MSR_BackFront = 2400/1000 #meter (y axis)
    MSR_TopBottom = 2400/1000 #meter (z axis)
    MSR_LeftRight = 2400/1000 #meter (x axis)
    
    axz.plot([-MSR_LeftRight/2, MSR_LeftRight/2, MSR_LeftRight/2,-MSR_LeftRight/2,-MSR_LeftRight/2],
             [ MSR_TopBottom/2, MSR_TopBottom/2,-MSR_TopBottom/2,-MSR_TopBottom/2, MSR_TopBottom/2],
            color=lcolor, linewidth=width)
    ayz.plot([-MSR_BackFront/2, MSR_BackFront/2, MSR_BackFront/2,-MSR_BackFront/2,-MSR_BackFront/2],
             [ MSR_TopBottom/2, MSR_TopBottom/2,-MSR_TopBottom/2,-MSR_TopBottom/2, MSR_TopBottom/2],
            color=lcolor, linewidth=width)
    axy.plot([-MSR_LeftRight/2, MSR_LeftRight/2, MSR_LeftRight/2,-MSR_LeftRight/2,-MSR_LeftRight/2],
             [ MSR_BackFront/2, MSR_BackFront/2,-MSR_BackFront/2,-MSR_BackFront/2, MSR_BackFront/2],
            color=lcolor, linewidth=width)

def DrawLayer6(axz,ayz,axy, lcolor = ["magenta","orchid"], width = 1):
    MSR_BackFront = 2207/1000 #meter (y axis)
    MSR_TopBottom = 2217/1000 #meter (z axis)
    MSR_LeftRight = 2217/1000 #meter (x axis)
    
    axz.plot([-MSR_LeftRight/2, MSR_LeftRight/2, MSR_LeftRight/2,-MSR_LeftRight/2,-MSR_LeftRight/2],
             [ MSR_TopBottom/2, MSR_TopBottom/2,-MSR_TopBottom/2,-MSR_TopBottom/2, MSR_TopBottom/2],
            color=lcolor[0], linewidth=width)
    ayz.plot([-MSR_BackFront/2, MSR_BackFront/2, MSR_BackFront/2,-MSR_BackFront/2,-MSR_BackFront/2],
             [ MSR_TopBottom/2, MSR_TopBottom/2,-MSR_TopBottom/2,-MSR_TopBottom/2, MSR_TopBottom/2],
            color=lcolor[0], linewidth=width)
    axy.plot([-MSR_LeftRight/2, MSR_LeftRight/2, MSR_LeftRight/2,-MSR_LeftRight/2,-MSR_LeftRight/2],
             [ MSR_BackFront/2, MSR_BackFront/2,-MSR_BackFront/2,-MSR_BackFront/2, MSR_BackFront/2],
            color=lcolor[0], linewidth=width)
    degaus_width = 0.02#meter, width of deagussing cable around edges
    axz.plot([-(MSR_LeftRight/2-degaus_width), (MSR_LeftRight/2-degaus_width), (MSR_LeftRight/2-degaus_width),-(MSR_LeftRight/2-degaus_width),-(MSR_LeftRight/2-degaus_width)],
             [ (MSR_TopBottom/2-degaus_width), (MSR_TopBottom/2-degaus_width),-(MSR_TopBottom/2-degaus_width),-(MSR_TopBottom/2-degaus_width), (MSR_TopBottom/2-degaus_width)],
            color=lcolor[1], linewidth=width*0.8,linestyle="--")
    ayz.plot([-(MSR_BackFront/2-degaus_width), (MSR_BackFront/2-degaus_width), (MSR_BackFront/2-degaus_width),-(MSR_BackFront/2-degaus_width),-(MSR_BackFront/2-degaus_width)],
             [ (MSR_TopBottom/2-degaus_width), (MSR_TopBottom/2-degaus_width),-(MSR_TopBottom/2-degaus_width),-(MSR_TopBottom/2-degaus_width), (MSR_TopBottom/2-degaus_width)],
            color=lcolor[1], linewidth=width*0.8,linestyle="--")
    axy.plot([-(MSR_LeftRight/2-degaus_width), (MSR_LeftRight/2-degaus_width), (MSR_LeftRight/2-degaus_width),-(MSR_LeftRight/2-degaus_width),-(MSR_LeftRight/2-degaus_width)],
             [ (MSR_BackFront/2-degaus_width), (MSR_BackFront/2-degaus_width),-(MSR_BackFront/2-degaus_width),-(MSR_BackFront/2-degaus_width), (MSR_BackFront/2-degaus_width)],
            color=lcolor[1], linewidth=width*0.8,linestyle="--")

def DrawCoruplast(axz,ayz,axy, lcolor = "teal", width = 1):
    MSR_BackFront = 48*25.4/1000 #meter (y axis)
    MSR_TopBottom = 48*25.4/1000 #meter (z axis)
    MSR_LeftRight = 48*25.4/1000 #meter (x axis)
    
    axz.plot([-MSR_LeftRight/2, MSR_LeftRight/2, MSR_LeftRight/2,-MSR_LeftRight/2,-MSR_LeftRight/2],
             [ MSR_TopBottom/2, MSR_TopBottom/2,-MSR_TopBottom/2,-MSR_TopBottom/2, MSR_TopBottom/2],
            color=lcolor, linewidth=width)
    ayz.plot([-MSR_BackFront/2, MSR_BackFront/2, MSR_BackFront/2,-MSR_BackFront/2,-MSR_BackFront/2],
             [ MSR_TopBottom/2, MSR_TopBottom/2,-MSR_TopBottom/2,-MSR_TopBottom/2, MSR_TopBottom/2],
            color=lcolor, linewidth=width)
    axy.plot([-MSR_LeftRight/2, MSR_LeftRight/2, MSR_LeftRight/2,-MSR_LeftRight/2,-MSR_LeftRight/2],
             [ MSR_BackFront/2, MSR_BackFront/2,-MSR_BackFront/2,-MSR_BackFront/2, MSR_BackFront/2],
            color=lcolor, linewidth=width)

