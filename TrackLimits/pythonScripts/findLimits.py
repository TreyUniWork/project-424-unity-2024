import numpy as np
from scipy import interpolate
from plotTrack import plot_autodata, plot_defineCorner, plot_limitFinding, plot_interpolation, plot_checkInside, plot_curvature

def loadTrackData(track_files):
    """_summary_

    Args:
        track_files (_type_): _description_

    Returns:
        _type_: _description_
    """
    nums = [str(i) for i in range(track_files[0], track_files[1])]

    segments = [np.genfromtxt("roadOutlines\\ROAD_GR"+n+".csv", dtype=None, delimiter=',') for n in nums]
    track = np.concatenate(segments, dtype=float)
    track, counts = np.unique(track, axis=0, return_counts=True)

    return track[counts<=4]

def findTrackLimits(track):
    """_summary_

    Args:
        track (_type_): _description_

    Returns:
        _type_: _description_
    """    
    sideR = [2,1]; sideL = [40,42]
    sides = []
    track_vertices = track[:,[0,2]]
    next_point = -1

    # while point does not = this point & no points remaining

    for i, side in enumerate([sideL, sideR]):
        while next_point != side[0]:
            # find all points within 3 metres, if less, try again

            next_point = find_next_point(track_vertices, i, side)

            side.append(next_point)
        sides.append(side)

    return sides

def find_next_point(track, side, previous, n=20):
    """_summary_

    Args:
        track (_type_): _description_
        side (_type_): _description_
        previous (_type_): _description_
        n (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """    
    # find next point in track limits

    prev = previous[-1]
    first = previous[0]

    dists = np.sum(np.square(track - track[prev]), axis=1)

    foundFirst = False

    dists[previous[1:]] = np.inf
    poss_points = np.argsort(dists)[1:1+n]

    if first in poss_points:
        first_ind = np.argwhere((poss_points == first))
        foundFirst = True

    b = track[poss_points] - track[prev]
    b = b.flatten()

    del_x = track[prev, 0] - track[previous[-2], 0]
    del_y = track[prev, 1] - track[previous[-2], 1]

    mag = np.sqrt(del_x**2 + del_y**2)
    parr = np.array([del_x, del_y]) / mag
    perp = np.array([-del_y, del_x]) / mag

    a = np.zeros((2*n,2*n))

    for i in range(n):
        a[i*2:i*2+2, i*2] = parr
        a[i*2:i*2+2, i*2+1] = perp

    arr = np.linalg.solve(a, b)
    parr = arr[::2]
    perp = arr[1::2]

    # magnitude = parr**2 + perp**2

    angle = np.arctan(perp/parr) * (1-2*side)

    angle[parr<0] = -np.inf

    angle[np.abs(angle)>np.pi/4] = -np.inf

    next_cand = np.argmax(angle)

    angle_diff = angle - angle[next_cand]

    angle[angle_diff < -(3*np.pi/180)] = -np.inf

    parr[angle == -np.inf] = np.inf

    next_cand = np.argmin(parr)

    if foundFirst and parr[first_ind] > 0 and parr[first_ind] != np.inf:
        return first
    else:
        return poss_points[next_cand]

def saveTrack(sides, track_name):
    """_summary_

    Args:
        sides (_type_): _description_
        track_name (_type_): _description_
    """
    with open("trackData\\"+track_name+"_sides.txt", 'w') as sideFile:

        sideFile.write(','.join([str(x) for x in sides[0]])+'\n' + ','.join([str(x) for x in sides[1]]))
        sideFile.close()

def loadTrack(track_name):
    """_summary_

    Args:
        track_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open("trackData\\track_sides\\"+track_name+"_sides.txt", 'r') as sideFile:
        
        sides = sideFile.read().split('\n')

        sideFile.close()

    return [[int(r) for r in x.split(',')] for x in sides]
    
def spline_curvature(tck, u, isSide = True):
    """_summary_

    Args:
        tck (_type_): _description_
        u (_type_): _description_
        side (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    x, y = interpolate.splev(u, tck, der=0)
    dx, dy = interpolate.splev(u, tck, der=1)
    ddx, ddy = interpolate.splev(u, tck, der=2)
    k = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** (3 / 2))

    psi = np.zeros(u.shape)

    mask_q1 = (dx>=0) & (dy>=0)
    mask_q2 = (dx<0) & (dy>=0)
    mask_q3 = (dx<0) & (dy<0)
    mask_q4 = (dx>=0) & (dy<0)

    psi[mask_q1] = np.arctan(dy[mask_q1]/dx[mask_q1])
    psi[mask_q2] = np.arctan(dy[mask_q2]/dx[mask_q2]) + np.pi
    psi[mask_q3] = np.arctan(dy[mask_q3]/dx[mask_q3]) + np.pi
    psi[mask_q4] = np.arctan(dy[mask_q4]/dx[mask_q4]) + np.pi*2

    mag = np.sqrt(dx**2+dy**2)
    
    if isSide:

        return np.array([u,x,y,k, dx/mag, dy/mag])

    else:

        return np.array([u,x,y,k,psi,u])

def interpolateTrack(track, sides):
    """_summary_

    Args:
        track (_type_): _description_
        sides (_type_): _description_

    Returns:
        _type_: _description_
    """
    track[:,0] *= -1
    track_vertices = track[:, [0,2]]

    #find closest point
    first = sides[0][0]
    dists = np.sum(np.square(track_vertices[sides[1]] - track_vertices[first]), axis=1)
    closest = np.argmin(dists)

    #reset the loop
    sides[1] = sides[1][:-1]
    sides[1] = sides[1][closest:] + sides[1][:closest] + [sides[1][closest]]

    tck_L, disc = interpolate.splprep(track_vertices[sides[0]].T, s=1e1)
    tck_R, disc = interpolate.splprep(track_vertices[sides[1]].T, s=1e1)

    temp_u = np.linspace(0,1,int(1e4))

    sideL = spline_curvature(tck_L,temp_u)
    sideR = spline_curvature(tck_R, temp_u)

    return sideL,sideR

def main():
    """_summary_
    """
    track_name = "nordschleife"
    track = loadTrackData((2,9))

    #sides = findTrackLimits(track)
    #saveTrack(sides, track_name)
    
    sides = loadTrack(track_name)

    sides = interpolateTrack(track, sides)

    auto_data = processAutopilot("trackData\\testPaths\\initial_path.csv", sides)

    #plot_autodata(auto_data, sides, track_name+" Corner Splitting")

    np.savetxt("trackData\\autopilot\\"+track_name + "_autopilot_interpolated.csv",auto_data, delimiter=',', newline='\n', fmt="%.5f")

    np.savetxt("trackData\\track_sides\\"+track_name+"_sidesDataL.csv", sides[0], delimiter=',', newline='\n')
    np.savetxt("trackData\\track_sides\\"+track_name+"_sidesDataR.csv", sides[1], delimiter=',', newline='\n')


def check_corner(start, end, u, k, min_peak = 1e-2, min_len = 20, min_rat=0.6):
    """_summary_

    Args:
        start (_type_): _description_
        end (_type_): _description_
        u (_type_): _description_
        k (_type_): _description_
        min_peak (_type_, optional): _description_. Defaults to 1e-2.
        min_len (int, optional): _description_. Defaults to 20.
        min_rat (float, optional): _description_. Defaults to 0.6.

    Returns:
        _type_: _description_
    """
    corner_inds = []

    for st, en in zip(start, end):

        k_t = k[st:en]
        u_t = u[st:en]

        ind_peak = np.argmax(k_t)
        peak = k_t[ind_peak]

        if peak > min_peak and en-st>min_len:

            before_inds = np.arange(1,ind_peak)
            after_inds = np.arange(ind_peak+1,k_t.size-1)

            before_u = u_t[before_inds]; before_k = k_t[before_inds]
            after_u = u_t[after_inds]; after_k = k_t[after_inds]

            before_del = (before_k-k_t[0]) / (before_u-u_t[0])
            after_del = (after_k-k_t[-1]) / (after_u-u_t[-1])

            bf_del_pk = (k_t[ind_peak]-k_t[0]) / (u_t[ind_peak]-u_t[0])
            af_del_pk = (k_t[ind_peak]-k_t[-1]) / (u_t[ind_peak]-u_t[-1])

            bf_i = np.nonzero((before_del<bf_del_pk*min_rat))[0]
            af_i = np.nonzero((after_del>af_del_pk*min_rat))[0]

            new_st = st
            if bf_i.size>0:
                st_new = before_inds[bf_i[-1]]
                roll_bf = np.arange(st_new,0,-1)
                # roll downwards
                delta_bf = np.ediff1d(k_t[roll_bf])
                deldel_bf = np.ediff1d(delta_bf)
                temp=np.nonzero((delta_bf[:-1]>0) | (deldel_bf>0))[0]
                if temp.size == 0:
                    st_new = roll_bf[-1]
                else:
                    st_new = roll_bf[temp[0]]
                new_st += st_new
            new_en = en
            if af_i.size>0:

                en_new = after_inds[af_i[0]]
                roll_af = np.arange(en_new,k_t.size,1)
                delta_af = np.ediff1d(k_t[roll_af])
                deldel_af = np.ediff1d(delta_af)
                temp=np.nonzero((delta_af[:-1]>0)|(deldel_af>0))[0]

                if temp.size == 0:
                    en_new = roll_af[-1]
                else:
                    en_new = roll_af[temp[0]]
                new_en = st+en_new

            #plot_defineCorner(u, k, st, en, new_st, new_en, st+ind_peak, min_rat, "TItle")

            if new_en-new_st > min_len:
                corner_inds.append([new_st,new_en])

    return corner_inds

def find_corners(u, k, threshold=1e-3):
    """_summary_

    Args:
        k (_type_): _description_
        threshold (_type_, optional): _description_. Defaults to 3e-3.

    Returns:
        _type_: _description_
    """
    #plot_curvature(u, k, threshold,"t")
    auto_corners = np.zeros(k.shape[0])

    k=np.abs(k)

    auto_corners[k > threshold] = 1

    mask = np.ediff1d(auto_corners)

    auto_corners[auto_corners == 0] = -1

    start = np.nonzero((mask>0))[0]+1
    end = np.nonzero((mask<0))[0]+1

    if auto_corners[0] == 1:
        start = np.concatenate((np.array([0]), start))
    if auto_corners[-1] == 1:
        end = np.concatenate((end, np.array([-2])))

    new_pairs = check_corner(start, end, u, k)    #en-st > 20 and en-st < 300

    auto_corners = np.zeros(k.shape[0]) -1

    for i, (st,en) in enumerate(new_pairs):

        auto_corners[st:en] = i

    return auto_corners.astype(int)

def processAutopilot(file_name, sides):
    """_summary_

    Args:
        file_name (_type_): _description_
        track_name (_type_): _description_
    """
    autopilot_data = np.genfromtxt(file_name, skip_header=2, delimiter=',', dtype=float)

    tck_auto, u_points = interpolate.splprep(autopilot_data[:,[10,12]].T, w=autopilot_data[:,20])
    auto_data = spline_curvature(tck_auto, u_points, False)

    #plot_interpolation(auto_data[0], auto_data[1:3], autopilot_data[:,[10,12]].T,sides, "X", "S")

    auto_data[-1] = find_corners(auto_data[0], auto_data[3])

    return auto_data

def AutopilotSine(auto_data, m=0.5, sigma=5e2):
    """_summary_

    Args:
        auto_data (_type_): _description_
        m (float, optional): _description_. Defaults to 0.5.
        sigma (_type_, optional): _description_. Defaults to 5e2.

    Returns:
        _type_: _description_
    """

    b_vector = auto_data[1:3,1:] - auto_data[1:3,:-1]

    perp_vector = auto_data[[6,5]]

    steering_angle = np.zeros(auto_data.shape[1])

    for i, (p_vec, b_temp) in enumerate(zip(perp_vector.T,b_vector.T)):

        a = np.array([p_vec[::-1], p_vec])
        parr, perp = np.linalg.solve(a, b_temp)

        steering_angle[i] = np.arctan(perp/parr) * (180/np.pi)

    return steering_angle


def check_inside(coords, sides, tol1=0.0, tol2 = -1, threshold=3e-3):
    """_summary_

    Args:
        coords (_type_): _description_
        sides (_type_): _description_
        tol1 (float, optional): _description_. Defaults to 0.0.
        tol2 (int, optional): _description_. Defaults to -1.
        threshold (_type_, optional): _description_. Defaults to 3e-3.

    Returns:
        _type_: _description_
    """

    #plot_checkInside(coords, sides, "title")
    for cd in coords.T:

        dist_l = np.sum(np.square(cd-sides[0][1:3].T),axis=1)
        dist_r = np.sum(np.square(cd-sides[1][1:3].T),axis=1)

        ind_min_l = np.argmin(dist_l); ind_min_r = np.argmin(dist_r)

        data_l=sides[0][:,ind_min_l]; data_r=sides[1][:,ind_min_r]
    
        delta_l = cd-data_l[1:3]; delta_r = cd-data_r[1:3]

        a_l = np.zeros((2,2)); a_r = np.zeros((2,2))
        vec_l = data_l[-2:]; vec_r = data_r[-2:]

        a_l[0] = vec_l; a_l[1] = vec_l[::-1]; a_l[1,1] *= -1
        a_r[0] = vec_r; a_r[1] = vec_r[::-1]; a_r[1,1] *= -1

        parr_l, perp_l = np.linalg.solve(a_l, delta_l)
        parr_r, perp_r = np.linalg.solve(a_r, delta_r)

        if -1*perp_l < tol1+(tol2-tol1)*(data_l[3]<threshold) or perp_r < tol1+(tol2-tol1)*(data_r[3]<threshold):
            
            return False

    return True

if __name__ == "__main__":
    """_summary_
    """    
    main()