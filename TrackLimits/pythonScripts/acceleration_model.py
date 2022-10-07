import numpy as np

def find_inputs(auto_data, c_inds):
    """_summary_

    Args:
        auto_data (_type_): _description_
        c_inds (_type_): _description_

    Returns:
        _type_: _description_
    """    
    data_len = auto_data.shape[1]
    lapdeltas = np.zeros(data_len)
    for i in range(lapdeltas.size):
        lapdeltas[1] = np.sqrt((auto_data[1, j+1]- auto_data[1, j])**2 + (auto_data[2, j+1]- auto_data[2, j])**2)


    speed = np.zeros(data_len)
    acc = np.zeros(data_len)

    mu =1.3
    g= 9.81
    speed[0] = 0
    acc[0]= mu*g

    for c_i in range(len(c_inds)):
        cornerentry = c_i[0]
        cornerconst = c_i[1]
        cornerexit = c_i[2]
        cornerfin = c_i[3]
        
        locentry = 0
        locconst = cornerconst - cornerentry
        locexit = cornerexit - cornerentry
        locfin = cornerfin- cornerentry
        
        cornerdata = auto_data[:, cornerentry:cornerfin+1]
        r = 1/ np.abs(cornerdata[3])
        deltas = np.zeros(len(r)-1)
        for j in range(len(r)-1):
            deltas[j] = np.sqrt((cornerdata[1, j+1]- cornerdata[1, j])**2 + (cornerdata[2, j+1]- cornerdata[2, j])**2)


        Uxconst = np.sqrt(mu*r[locconst]*g)
        
        Uxexit = np.zeros(locfin-locexit+1)
        Uxexit[0] = Uxconst
        axexit = np.zeros(locfin-locexit)
        Rexit = r[locexit-1:locfin]
        deltasexit = deltas[locexit-1:locfin-1]

        for j in range(len(axexit)):
            if j == 0:
                c = np.sqrt(1/(2*(deltasexit[j])*Rexit[j]))
                if (mu*g)**2 - (2*c**2*(deltasexit[j])*Uxexit[j]**2)**2 < 0:
                    Uxexit[j+1] = Uxexit[j]
                else:
                    Uxexit[j+1] = Uxexit[j] + (deltasexit[j]/Uxexit[j])*np.sqrt((mu*g)**2 - (2*c**2*(deltasexit[j])*Uxexit[j]**2)**2)
                
                c = np.sqrt(1/(2*(np.sum(deltasexit[0:j+1]))*Rexit[j+1]))   
                if (mu*g)**2- (2*c**2 * (np.sum(deltasexit[0:j+1]))*Uxexit[j+1]**2)**2 < 0:
                    axexit[j] = 1
                else:
                    axexit[j] = np.sqrt((mu*g)**2- (2*c**2 * (np.sum(deltasexit[0:j+1]))*Uxexit[j+1]**2)**2)  
                
            else:
                c = np.sqrt(1/(2*(np.sum(deltasexit[0:j]))*Rexit[j])) 
                if (mu*g)**2 - (2*c**2 *( np.sum(deltasexit[0:j]))*Uxexit[j]**2)**2< 0:
                    Uxexit[j+1] =Uxexit[j]
                else:
                    Uxexit[j+1] = Uxexit[j] + (deltasexit[j]/Uxexit[j])*np.sqrt((mu*g)**2 - (2*c**2 *(np.sum(deltasexit[0:j]))*Uxexit[j]**2)**2)
                c = np.sqrt(1/(2*( np.sum(deltasexit[0:j+1]))*Rexit[j+1]))
                if (mu*g)**2- (2*c**2 * ( np.sum(deltasexit[0:j+1]))*Uxexit[j+1]**2)**2 < 0:
                    axexit[j] = 1
                else:
                    axexit[j] = np.sqrt((mu*g)**2- (2*c**2 * (np.sum(deltasexit[0:j+1]))*Uxexit[j+1]**2)**2)

        speed[range(cornerexit, cornerfin)] = Uxexit[range(1, len(Uxexit))]
        acc[range(cornerexit, cornerfin)] = axexit
        
        
        speed[range(cornerconst,cornerexit)] = Uxconst
        acc[range(cornerconst,cornerexit)] = 1
        
        
        Uxentry = np.zeros(locconst + 1)
        Uxentry[locconst] = Uxconst
        axentry = np.zeros(locconst)
        Rentry = r[:locconst +1]
        deltasentry = deltas[:locconst+1]
        
        for j in range(locconst, 0, -1):
            if j == locconst:
                    c = np.sqrt(1/(2*(deltasentry[j])*Rentry[j]))
                    if (mu*g)**2 - (2*c**2*(deltasentry[j])*Uxentry[j]**2)**2 < 0:
                        Uxentry[j-1] = Uxentry[j]
                    else:
                        Uxentry[j-1] = Uxentry[j] + (deltasentry[j]/Uxentry[j])*np.sqrt((mu*g)**2 - (2*c**2*(deltasentry[j])*Uxentry[j]**2)**2) 

            else:
                c = np.sqrt(1/(2*(np.sum(deltasentry[j:locconst]))*Rentry[j]))
                if (mu*g)**2 - (2*c**2 *( np.sum(deltasentry[j:locconst]))*Uxentry[j]**2)**2< 0:
                    Uxentry[j-1] = Uxentry[j]
                else:
                    Uxentry[j-1] = Uxentry[j] + (deltasentry[j]/Uxentry[j])*np.sqrt((mu*g)**2 - (2*c**2 *(np.sum(deltasentry[j:locconst]))*Uxentry[j]**2)**2)

        for j in range(len(axentry)):
            c = np.sqrt(1/(2*( np.sum(deltasentry[0:j+1]))*Rentry[j+1]))
            if (mu*g)**2- (2*c**2 * ( np.sum(deltasentry[0:j+1]))*Uxentry[j+1]**2)**2 < 0:
                axentry[j] = 1
            else:
                axentry[j] = np.sqrt((mu*g)**2- (2*c**2 * (np.sum(deltasentry[0:j+1]))*Uxentry[j+1]**2)**2)
        #print(axentry)

        speed[range(cornerentry, cornerconst)] = Uxentry[range(len(Uxentry)-1)]
        acc[range(cornerentry, cornerconst)] = -axentry
        
        if i == 0:
            for j in range(cornerentry):
                if np.sum(lapdeltas[range(j, cornerentry)]) > (speed[j]**2 - Uxentry[0]**2)/(2*mu*g):
                    acc[j+1] = mu*g
                    speed[j+1] = np.sqrt(speed[j]**2 + 2*acc[j]*lapdeltas[j])
                else:
                    acc[j+1] = -mu*g
                    speed[j+1] = np.sqrt(speed[j]**2 + 2*acc[j]*lapdeltas[j])
                                    
        else:
            for j in range(c_inds[i-1, 3], cornerentry):
                if np.sum(lapdeltas[range(j, cornerentry)]) > (speed[j]**2 - Uxentry[0]**2)/(2*mu*g):
                    acc[j+1] = mu*g
                    speed[j+1] = np.sqrt(speed[j]**2 + 2*acc[j]*lapdeltas[j])
                else:
                    acc[j+1] = -mu*g
                    speed[j+1] = np.sqrt(speed[j]**2 + 2*acc[j]*lapdeltas[j])

        if i == len(c_inds)-1:
            for j in range(c_inds[i, 3], len(auto_data.columns)-1):
                acc[j+1] = mu*g
                speed[j+1] = np.sqrt(speed[j]**2 + 2*acc[j]*lapdeltas[j])
                
                
    throttle = np.zeros(len(speed))
    brake = np.zeros(len(speed))
    for i in range(len(speed)):
        if acc[i] == mu*g:
            throttle[i] = 100
            brake[i] = 0
        elif acc[i] >= 0:
            throttle[i] = min(0.2873 * speed[i] + 4.1378*acc[i], 100)
            brake[i] = 0
        else:
            throttle[i] = 0
            brake[i] = min(-0.0531*speed[i]  -2.8267*acc[i], 100)

    return np.array([throttle, brake])
