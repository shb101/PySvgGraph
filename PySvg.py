import numpy as np
import time
from BezierCurveFitting import BezierCurveFittingModel

class SvgPlot:
    def __init__(self, width, height, margin_width, margin_height, x_min, x_max, y_min, y_max):
        # Initialize parameters:
        self.W = width
        self.H = height
        self.mW = margin_width
        self.mH = margin_height
        self.x0 = x_min
        self.x1 = x_max
        self.y0 = y_min
        self.y1 = y_max
        self.map = lambda x, y: (
            (self.W - 2*self.mW)/(self.x1 - self.x0) * (x-self.x0) + self.mW,
            -(self.H - 2*self.mH)/(self.y1 - self.y0) * (y-self.y0) + self.H - self.mH
        )
        #self.filetxt = f'''<svg width="{self.W}" height="{self.H}" xmlns="http://www.w3.org/2000/svg">\n'''
        self.filetxt = ''

    def addLinePath(self, points, color='black', thickness=2):
        mappedPoints = []
        for (x,y) in points:
            mappedPoints.append(self.map(x,y))
        # Create the path:
        (x0,y0) = mappedPoints.pop(0)
        pathToAdd = f'M{x0} {y0} '
        for (x,y) in mappedPoints:
            pathToAdd += f'L{x} {y} '
        # Add the path:
        self.filetxt += f'''\t<path d="{pathToAdd}" fill-opacity="0" stroke="{color}" stroke-width="{thickness}"/>\n'''
    
    def addCubicPath(self, points, color='black', thickness=2):
        mappedPoints = []
        for (P0, P1, P2, P3) in points:
            mappedPoints.append(
                (self.map(P0[0], P0[1]), self.map(P1[0], P1[1]), self.map(P2[0], P2[1]), self.map(P3[0], P3[1]))
            )
        # Create the path:
        pathToAdd = f'M {mappedPoints[0][0][0]} {mappedPoints[0][0][1]} '
        for (_, P1, P2, P3) in mappedPoints:
            pathToAdd += f'C {P1[0]} {P1[1]}, {P2[0]} {P2[1]}, {P3[0]} {P3[1]} '
        # Add the path:
        self.filetxt += f'''\t<path d="{pathToAdd}" fill-opacity="0" stroke="{color}" stroke-width="{thickness}"/>\n'''
    
    def drawAxes(self, xlabel=None, ylabel=None, color='darkgray', thickness=2, ticks='auto'):
        # Settings
        l = 5 # axis extra len
        s = 75 # tick spacing
        t = 10 # tick size
        dx = (self.x1 - self.x0)/(self.W - 2*self.mW)
        dy = (self.y1 - self.y0)/(self.H - 2*self.mH)
        # Draw x and y axes
        self.drawLine(self.x0, self.y0-l*dy, self.x0, self.y1+l*dy, color, thickness)
        self.drawLine(self.x0-l*dx, self.y0, self.x1+l*dx, self.y0, color, thickness)
        if xlabel is not None:
            self.addText((self.x0+self.x1)/2, self.y0-dy*t*4, xlabel)
        if ylabel is not None:
            self.addText(self.x0-dx*t*5, (self.y0+self.y1)/2, ylabel, rotate=270)
        if ticks == 'auto':
            # x ticks:
            self.addText(self.x0, self.y0-dy*t*2, f'{self.x0:.2f}')
            i = 1
            while self.x0 + i*dx*s < self.x1:
                self.drawLine(self.x0+i*dx*s, self.y0-dy*t/2, self.x0+i*dx*s, self.y0+dy*t/2, color, thickness)
                self.addText(self.x0+i*dx*s, self.y0-dy*t*2, f'{self.x0+i*dx*s:.2f}')
                i += 1
            # y ticks:
            self.addText(self.x0-dx*t, self.y0, f'{self.y0:.2f}', anchor='end')
            i = 1
            while self.y0 + i*dy*s < self.y1:
                self.drawLine(self.x0-dx*t/2, self.y0+i*dy*s, self.x0+dx*t/2, self.y0+i*dy*s, color, thickness)
                self.addText(self.x0-dx*t, self.y0+i*dy*s, f'{self.y0+i*dy*s:.2f}', anchor='end')
                i += 1
        
    def fitBezierCurves(self, f, n):
        # n is the number of bezier curves to fit
        model = BezierCurveFittingModel(f,n,self.x0,self.x1)
        points = model.solve()
        return points
    
    def drawLine(self, x1, y1, x2, y2, color='black', thickness=2):
        (x1, y1) = self.map(x1, y1)
        (x2, y2) = self.map(x2, y2)
        self.filetxt += f'''\t<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{thickness}"/>\n'''
    
    def addText(self, x, y, txt, color='black', rotate=0, anchor='middle'):
        (x, y) = self.map(x, y)
        if rotate != 0:
            self.filetxt += f'''\t<text transform="translate({x}, {y}) rotate({rotate})" text-anchor="{anchor}" dominant-baseline="middle">{txt}</text>\n'''
        else:
            self.filetxt += f'''\t<text x="{x}" y="{y}" fill="{color}" text-anchor="{anchor}" dominant-baseline="middle">{txt}</text>\n'''

    def linePlot(self, x, y, color='black', thickness=2):
        points = [(x[i], y[i]) for i in range(len(x))]
        self.addLinePath(points, color, thickness)

    def functionPlot(self, f, n, color='black', thickness=2):
        # Requires 5*n function evaluations and n gradient descent runs to compute cubic bezier control points 
        points = self.fitBezierCurves(f, n)
        self.addCubicPath(points, color, thickness)

    def save(self, filename):
        txtToFile = f'''<svg width="{self.W}" height="{self.H}" xmlns="http://www.w3.org/2000/svg">\n''' + self.filetxt + '</svg>'
        with open(filename, 'w') as f:
            f.write(txtToFile)

if __name__ == '__main__':
    time1 = time.time()
    img = SvgPlot(width=800, height=500, margin_width=70, margin_height=50, x_min=0, x_max=4*np.pi, y_min=-1, y_max=1)
    img.functionPlot(lambda x: np.sin(x), n=10)
    img.functionPlot(lambda x: np.cos(x), n=10, color='red')

    #x = np.linspace(0,4*np.pi,100)
    #y = np.sin(x)
    #img.linePlot(x,y)
    #y = np.cos(x)
    #img.linePlot(x,y, color='red')

    f = lambda x: np.sin(x)/x if x != 0 else 1
    img.functionPlot(f, n=12, color='blue')

    g = lambda x: np.cos(x)/(1+x)
    img.functionPlot(g, n=12, color='green')

    img.drawAxes(xlabel='x axis', ylabel='y axis')
    img.save('sinwave.svg')
    time2 = time.time()
    print(f'Time: {time2-time1:.3f} seconds')
