#v2_4，相比于He2021_v2_3.py的更新：

#1. 加入了（改进了）计算timedelay的三个函数（TwoImgTimeDelay, FourImgTimeDelay, TimeDelay)

#2021-5-26 zizhao He, NAOC

#2. 改进了AddPoiNoise,使得输入输出一致

#2021-6-27 zizhao He, NAOC

#3. 添加FindIdxForGivenRange

#2021-7-14 zizhao He, NAOC

#4. 往GetSourcePosition添加return (x_rand[0], y_rand[0])

#2023-6-15 zizhao He, NAOC

#5. 加入numba_flag，可以开关numba-nopython模式

#已知问题：现在开启numba-nopython时会报错。

import math
import random as r
import numpy as np
from numba import jit
from scipy import ndimage
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from shapely.geometry import Point
import lenstronomy.Util.constants as c
from shapely.geometry.polygon import Polygon
from astropy.cosmology import Planck18 as cosmo
numba_flag=False

def AddGusPSF(oriimg, sigma):
    #只能用于正方形
    oriimg = fftpack.fftshift(oriimg)
    length,_ = oriimg.shape
    
    m = length/2
    x = np.linspace(1,length,length)
    y = np.linspace(1,length,length)
    [X, Y] = np.meshgrid(x,y)
    kernelimage = (1/(2*np.pi*sigma**2))*np.exp(-((X-m)**2 + (Y-m)**2)/(2*sigma**2))

    fftimg = fftpack.fft2(oriimg)
    fftkernel = fftpack.fft2(kernelimage)
    fftkernel[fftkernel == 0] = 1e-20
    fftblurimage = fftimg*fftkernel
    blurimage = np.real(fftpack.ifft2(fftblurimage))
    
    return blurimage

def AddPoiNoise(ori_img, which_band):
    length_of_img = ori_img.shape[0] #假设输入图像是正方形

    background_cut_x = 0.25
    background_cut_y = 0.25

    upper_left_xrange = np.linspace(0, (background_cut_x * length_of_img - 1)).astype(int)
    upper_left_yrange = upper_left_xrange
    lower_left_xrange = upper_left_xrange
    lower_left_yrange = np.linspace((1-background_cut_x)*length_of_img-1,length_of_img-1).astype(int)
    upper_right_xrange = lower_left_yrange
    upper_right_yrange = upper_left_yrange
    lower_right_xrange = lower_left_yrange
    lower_right_yrange = lower_right_xrange

    upper_left_xmin = upper_left_xrange[0]
    upper_left_xmax = upper_left_xrange[-1]
    upper_left_ymin = upper_left_yrange[0]
    upper_left_ymax = upper_left_yrange[-1]

    lower_left_xmin = lower_left_xrange[0]
    lower_left_xmax = lower_left_xrange[-1]
    lower_left_ymin = lower_left_yrange[0]
    lower_left_ymax = lower_left_yrange[-1]

    upper_right_xmin = upper_right_xrange[0]
    upper_right_xmax = upper_right_xrange[-1]
    upper_right_ymin = upper_right_yrange[0]
    upper_right_ymax = upper_right_yrange[-1]

    lower_right_xmin = lower_right_xrange[0]
    lower_right_xmax = lower_right_xrange[-1]
    lower_right_ymin = lower_right_yrange[0]
    lower_right_ymax = lower_right_yrange[-1]

    ori_img_cutted_1 = ori_img[upper_left_xmin:upper_left_xmax, upper_left_ymin:upper_left_ymax]
    ori_img_cutted_2 = ori_img[upper_right_xmin:upper_right_xmax, upper_right_ymin:upper_right_ymax]
    ori_img_cutted_3 = ori_img[lower_left_xmin:lower_left_xmax, lower_left_ymin:lower_left_ymax]
    ori_img_cutted_4 = ori_img[lower_right_xmin:lower_right_xmax,lower_right_ymin:lower_right_ymax]
    
    mean_arr = np.array([ori_img_cutted_1.mean(), ori_img_cutted_2.mean(), ori_img_cutted_3.mean(), ori_img_cutted_4.mean()])
    which_is_min = np.argmin(mean_arr) + 1
    ori_img_cut_min = locals()['ori_img_cutted_'+str(which_is_min)]
    poi_sky_std = ori_img_cut_min.std()

    poi_sky = np.random.randn(length_of_img, length_of_img) * poi_sky_std

    expst_g = 166
    expst_r = 134
    expst_z = 200
    
    if which_band == 'g':
        img_n = np.sqrt(abs(ori_img)/expst_g)
    elif which_band == 'r':
        img_n = np.sqrt(abs(ori_img)/expst_r)
    elif which_band == 'z':
        img_n = np.sqrt(abs(ori_img)/expst_z)
    else:
        print('No proper band info. was provided.')
        return False

    poi_src = np.random.randn(length_of_img, length_of_img) * img_n
    
    img_noisy = ori_img + poi_src + poi_sky
    
#     img_noisy = img_noisy*(ori_img.sum()/img_noisy.sum())
    
    return img_noisy

def Appmag2Sigmav(zlens, zsource, re_arcsec, flux_r):
    a_par = 1.4335; b_par = 0.3150; c_par = -8.8979

    Dd_angle = cosmo.angular_diameter_distance(zlens).value                     #Mpc
    Ds_angle = cosmo.angular_diameter_distance(zsource).value                   #Mpc
    Dds_angle = cosmo.angular_diameter_distance_z1z2(zlens,zsource).value       #Mpc

    re_rad = (re_arcsec/3600) * np.pi / 180
    re_kpc = re_rad * Dd_angle * 1000                                           #kpc

    flux_r_corrected = flux_r * 3.97 / 2
    m_r_corrected = 22.5 - 2.5 * np.log10(flux_r_corrected)
    m_r = 22.5 - 2.5 * np.log10(flux_r)

    miu_e = m_r + 5 * np.log10(re_arcsec) + 2.5 * np.log10(2 * np.pi) - 10 * np.log10(1+zlens)
    sigma = 10** ((1/a_par) * (np.log10(re_kpc) - b_par * miu_e - c_par)) * 1000 #SI
    sigma = sigma * np.random.normal(1, 0.1 , [1, 1])
    
    return sigma

@jit(nopython=numba_flag)
def DefSIE(x, y, lpar_local):
    # Calculating the img coordinate in source plane of an SIE mass profile following Kormann 1993
    # The convergence has the form of kappa(x, y)=0.5*sqrt(q)*b_sie/sqrt(x^2+q^2*y^2)
    # In this form, b_sie is the Einstein radius in the intermediate-axis convention
    # lpar[0]: lens einstein radius arcsec
    # lpar[1]: lens xcenter
    # lpar[2]: lens ycenter
    # lpar[3]: position angle in degreef
    # lpar[4]: axis ratio
    # lpar[3] = lpar[3]-90

    if lpar_local[4] > 1.0:
        lpar_local[4] = 1.0 / lpar_local[4]
        lpar_local[3] = lpar_local[3] + 90.0
    if lpar_local[3] > 180.0:
        lpar_local[3] = lpar_local[3] - 180.0
    elif lpar_local[3] < 0.0:
        lpar_local[3] = lpar_local[3] + 180.0
    (xnew, ynew) = XYTransform(x, y, lpar_local[1], lpar_local[2], lpar_local[3])  # rotational matrix

    r_sie = np.sqrt(xnew ** 2. + ynew ** 2.)
    qfact = np.sqrt((1.0 / lpar_local[4] - lpar_local[4]))
    eps = 10. ** (-8)
    if np.abs(qfact) <= eps:  # sie -> sis
        alpha_x = xnew / (r_sie + (r_sie == 0))
        alpha_y = ynew / (r_sie + (r_sie == 0))
    else:
        alpha_x = np.arcsinh(np.sqrt(1.0 / lpar_local[4] ** 2.0 - 1.0) * xnew / (r_sie + (r_sie == 0))) / qfact
        alpha_y = np.arcsin(np.sqrt(1.0 - lpar_local[4] ** 2.0) * ynew / (r_sie + (r_sie == 0))) / qfact
    (alpha_x_new, alpha_y_new) = XYTransform(alpha_x, alpha_y, 0.0, 0.0, -lpar_local[3])

    src_x = x - lpar_local[0] * alpha_x_new
    src_y = y - lpar_local[0] * alpha_y_new

    return (src_x, src_y)

def DropCntrImg(tri_img_plane,index_satisfied,pixel_size):

    p1 = [pixel_size/4,pixel_size/4]
    p2 = [pixel_size/4,-pixel_size/4]
    p3 = [-pixel_size/4,pixel_size/4]
    p4 = [-pixel_size/4,-pixel_size/4]
    p5 = [pixel_size*3/4,pixel_size*3/4]
    p6 = [pixel_size*3/4,-pixel_size*3/4]
    p7 = [-pixel_size*3/4,pixel_size*3/4]
    p8 = [-pixel_size*3/4,-pixel_size*3/4]

    for idx in index_satisfied:
        this_tri = tri_img_plane[idx]
        j1 = IsInside(this_tri[0],this_tri[1],this_tri[2],p1)
        j2 = IsInside(this_tri[0],this_tri[1],this_tri[2],p2)
        j3 = IsInside(this_tri[0],this_tri[1],this_tri[2],p3)
        j4 = IsInside(this_tri[0],this_tri[1],this_tri[2],p4)
        j5 = IsInside(this_tri[0],this_tri[1],this_tri[2],p5)
        j6 = IsInside(this_tri[0],this_tri[1],this_tri[2],p6)
        j7 = IsInside(this_tri[0],this_tri[1],this_tri[2],p7)
        j8 = IsInside(this_tri[0],this_tri[1],this_tri[2],p8)

        if j1 or j2 or j3 or j4 or j5 or j6 or j7 or j8:
            index_satisfied.remove(idx) 

    return(index_satisfied)

def FermatPtntl(theta_x, theta_y, lpar_local):
    #according to N.Li + 2020, eq.(7)
    #https://arxiv.org/pdf/2006.08540.pdf
    
    #term1
    beta_x, beta_y = DefSIE(theta_x, theta_y, lpar_local)
    term1 = 0.5 * ((theta_x-beta_x)**2+(theta_y-beta_y)**2)
    
    #term2
    term2 = KormannPtntl(theta_x, theta_y, lpar_local)
    
    return term1, term1-term2

def FindIdxForGivenRange(array, width):
    lowwer = width[0]; upper = width[1]
    comidx_range1 = np.where(array > lowwer)
    comidx_range2 = np.where(array < upper)
    comidx = np.intersect1d(comidx_range1, comidx_range2)
    
    return comidx

def FindTriangleShareOneCommonSide(A, v1, v2, index_focus):
    t1, t2 = zip(*(A - v1).reshape(int(A.size / 2), 2));
    t3, t4 = zip(*(A - v2).reshape(int(A.size / 2), 2))
    t5 = np.argwhere(np.array(t1) == 0)
    t6 = np.argwhere(np.array(t2) == 0)
    t7 = np.argwhere(np.array(t3) == 0)
    t8 = np.argwhere(np.array(t4) == 0)
    t9 = np.intersect1d(t5, t6)
    t10 = np.intersect1d(t7, t8)
    common_indexs = np.intersect1d((t9 // 3).astype(int), (t10 // 3).astype(int))
    return np.setdiff1d(common_indexs, [index_focus])

def FindNearbyTriangles(A, index_focus, typ):
    v1 = A[index_focus, 0];
    v2 = A[index_focus, 1];
    v3 = A[index_focus, 2];
    fisrt_tri_index = FindTriangleShareOneCommonSide(A, v1, v2, index_focus)
    second_tri_index = FindTriangleShareOneCommonSide(A, v1, v3, index_focus)
    thrid_tri_index = FindTriangleShareOneCommonSide(A, v2, v3, index_focus)

    if typ == 'self_included':
        nearby_tri_indexs = np.array([fisrt_tri_index, second_tri_index, thrid_tri_index, np.array([index_focus])])
    else:
        nearby_tri_indexs = np.array([fisrt_tri_index, second_tri_index, thrid_tri_index])

    nearby_tri_indexs = nearby_tri_indexs.reshape(1, -1)

    _, length = nearby_tri_indexs.shape
    index_empty = []

    for idx in range(length):
        if nearby_tri_indexs[0][idx].tolist() == []:
            index_empty.append(idx)

    nearby_tri_indexs = np.delete(nearby_tri_indexs, index_empty, 1)
    nearby_tri_indexs = nearby_tri_indexs.reshape(1, -1)
    nearby_tri_indexs = nearby_tri_indexs.astype(int)
    nearby_tris = A[nearby_tri_indexs].reshape(-1, 3, 2)

    return nearby_tris

def FourImgTimeDelay(img_ra_dac_sort_arr, lpar_local, zd,zs):
    
    if len(img_ra_dac_sort_arr)!=8:
        print("Require Four Imgs, but ", len(img_ra_dac_sort_arr)/2," are given.")
        return False
    else:
        theta_A_x = img_ra_dac_sort_arr[0]
        theta_A_y = img_ra_dac_sort_arr[1]
        theta_B_x = img_ra_dac_sort_arr[2]
        theta_B_y = img_ra_dac_sort_arr[3]
        theta_C_x = img_ra_dac_sort_arr[4]
        theta_C_y = img_ra_dac_sort_arr[5]
        theta_D_x = img_ra_dac_sort_arr[6]
        theta_D_y = img_ra_dac_sort_arr[7]
        
        _,Fermat_A = FermatPtntl(theta_A_x, theta_A_y, lpar_local)
        _,Fermat_B = FermatPtntl(theta_B_x, theta_B_y, lpar_local)
        _,Fermat_C = FermatPtntl(theta_C_x, theta_C_y, lpar_local)
        _,Fermat_D = FermatPtntl(theta_D_x, theta_D_y, lpar_local)

        deltaFermat_AB = Fermat_A - Fermat_B
        deltaFermat_AC = Fermat_A - Fermat_C
        deltaFermat_AD = Fermat_A - Fermat_D

        Dd =  cosmo.angular_diameter_distance(zd).si.value               #m
        Ds =  cosmo.angular_diameter_distance(zs).si.value               #m
        Dds = cosmo.angular_diameter_distance_z1z2(zd,zs).si.value       #m

        t_AB = (1/c.day_s)*(1/c.c)*((1+zd)*(Dd*Ds)/Dds)*deltaFermat_AB   #day*arcsec^2
        t_AB = t_AB * c.arcsec ** 2                                      #day
        
        t_AC = (1/c.day_s)*(1/c.c)*((1+zd)*(Dd*Ds)/Dds)*deltaFermat_AC   #day*arcsec^2
        t_AC = t_AC * c.arcsec ** 2                                      #day
        
        t_AD = (1/c.day_s)*(1/c.c)*((1+zd)*(Dd*Ds)/Dds)*deltaFermat_AD   #day*arcsec^2
        t_AD = t_AD * c.arcsec ** 2                                      #day
        
        t_A = 0; t_B = t_A - t_AB; t_C = t_A - t_AC; t_D = t_A - t_AD;
        t_earliest = min(t_A, t_B, t_C, t_D)
        t_A = t_A - t_earliest; t_B = t_B - t_earliest; t_C = t_C - t_earliest; t_D = t_D - t_earliest;
        
        return t_A, t_B, t_C, t_D

@jit(nopython=numba_flag)
def GetArea(vertex1, vertex2, vertex3):
    x1 = vertex1[0]
    y1 = vertex1[1]
    x2 = vertex2[0]
    y2 = vertex2[1]
    x3 = vertex3[0]
    y3 = vertex3[1]

    return np.sqrt(((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)**2)

@jit(nopython=numba_flag)
def GetBaryCoor(triangle):
    x1 = triangle[0][0]
    y1 = triangle[0][1]
    x2 = triangle[1][0]
    y2 = triangle[1][1]
    x3 = triangle[2][0]
    y3 = triangle[2][1]
    return [(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3]

def GetCritialCastic(x,y, miu_inversed_Kormann,lpar_local):
    
    tmpfig = plt.figure(); tmpsubplot = tmpfig.add_subplot(111)
    
    cset_critical_Kormann = tmpsubplot.contour(x, y, miu_inversed_Kormann, [0.])
    cset_critical_Kormann = cset_critical_Kormann.collections[0].get_paths()
    
    plt.close(tmpfig)
    
    critical_Kormann =[]
    
    for contour in cset_critical_Kormann:
        xcon,ycon = contour.vertices[:,0], contour.vertices[:,1]
        critical_Kormann.append(np.vstack([xcon,ycon]).T)
    
    critical_Kormann_inner = critical_Kormann[1]
    critical_Kormann_outer = critical_Kormann[0]

    critical_Kormann_inner = np.array(critical_Kormann_inner).reshape(-1,2)
    critical_Kormann_outer = np.array(critical_Kormann_outer).reshape(-1,2)

    caustic_Kormann = []
    for theta_x, theta_y in critical_Kormann_outer:
        caustic_Kormann.append(DefSIE(theta_x, theta_y, lpar_local))
        
    caustic_Kormann_psdo = []
    for theta_x, theta_y in critical_Kormann_inner:
        caustic_Kormann_psdo.append(DefSIE(theta_x, theta_y, lpar_local))
        
    return critical_Kormann_inner, critical_Kormann_outer, caustic_Kormann, caustic_Kormann_psdo

def GetDistance(p1, p2):
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

def GetFluxOfImg(miu, mag):
    #22.5是decals的仪器零点星等
    oriflux_counts_per_sec = 10**((-1.0/2.50)*(mag-22.5))
    imgflux_counts_per_sec = miu * oriflux_counts_per_sec
    
    return imgflux_counts_per_sec

def GetImgPlaneWithFlux(plane_xx, plane_yy, miu_list, img_ra_dac_list,mag_QSO, scale):
    img_plane = np.zeros([plane_xx.shape[0],plane_xx.shape[1]])
    num_of_imgs = len(miu_list)
    for i in range(num_of_imgs):
        x_foucs,y_foucs = img_ra_dac_list[i]
        x_indx = int(round(((x_foucs+scale)/(2*scale)) * plane_xx.shape[0]))
        y_indx = int(round(((y_foucs+scale)/(2*scale)) * plane_yy.shape[1]))
        img_plane[x_indx][y_indx] = GetFluxOfImg(miu_list[i], mag_QSO)
    return img_plane

def GetImgPositionOfOneImg(A, index_foucs, lpar_local, beta_ra_dec):
    if IsInside(A[index_foucs][0], A[index_foucs][1], A[index_foucs][2], [0, 0]):
        return [1e-8, 1e-8], np.array(0)

    else:

        QSO_bary_distance = 1

        run_num = 0

        bary_crdnt_list = []
        bary_move = 1

        while bary_move > 1e-6:
            lager_tri_img_plane_with_nearby_tirs = FindNearbyTriangles(A, index_foucs, 'self_included')

            lager_tri_img_plane_with_nearby_tirs_segemented = \
                SplitingTriangles(lager_tri_img_plane_with_nearby_tirs)

            lager_tri_source_plane_with_nearby_tirs_segemented = \
                TriangleImg2Source(lager_tri_img_plane_with_nearby_tirs_segemented, lpar_local)

            # 找到符合条件的小三角形
            index_satisfied = GetIndexOfSatisfiedTriangle(lager_tri_source_plane_with_nearby_tirs_segemented,
                                                          beta_ra_dec)

            if len(index_satisfied)!=1:
                print("Warning: len(index_satisfied)",len(index_satisfied))
    
                if len(index_satisfied)== 0:
                    return ('Catch a bug', 'Catch a bug'), 'Catch a bug'
                else:
                    index_satisfied = [index_satisfied[0]]
            
            lager_tri_source_plane_with_nearby_tirs_segemented_satisfied = \
                lager_tri_source_plane_with_nearby_tirs_segemented[index_satisfied].reshape(3, 2)

            lager_tri_img_plane_with_nearby_tirs_segemented_satisfied = \
                lager_tri_img_plane_with_nearby_tirs_segemented[index_satisfied].reshape(3, 2)

            lager_tri_source_plane_with_nearby_tirs_segemented_satisfied_bary = \
                GetBaryCoor(lager_tri_source_plane_with_nearby_tirs_segemented_satisfied)

            QSO_bary_distance = GetDistance \
                (lager_tri_source_plane_with_nearby_tirs_segemented_satisfied_bary, beta_ra_dec)

            A = lager_tri_img_plane_with_nearby_tirs_segemented

            index_foucs = index_satisfied[0]

            bary_crdnt_list.append(lager_tri_source_plane_with_nearby_tirs_segemented_satisfied_bary)

            if run_num>0:
                bary_move = GetDistance(bary_crdnt_list[run_num],bary_crdnt_list[run_num-1])

            run_num += 1

        return \
            GetBaryCoor(lager_tri_img_plane_with_nearby_tirs_segemented_satisfied), run_num

@jit(nopython=numba_flag)
def GetIndexOfSatisfiedTriangle(tri, beta_ra_dec):
    
#     print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOK")
    index_satisfied = []
    
    for index in range(len(tri)):
        vertex1 = tri[index][0]
        vertex2 = tri[index][1]
        vertex3 = tri[index][2]
        
        if IsInside(vertex1, vertex2, vertex3, beta_ra_dec):
            index_satisfied.append(index)
        index += 1

    return index_satisfied

def GetInversedMiu(x,y, pixel_size, lpar_local):
    
    Psi_Kormann = KormannPtntl(x, y, lpar_local)
    
    Psi_Kormann_y, Psi_Kormann_x = np.gradient(Psi_Kormann)/np.array(pixel_size)
    Psi_Kormann_xy, Psi_Kormann_xx = np.gradient(Psi_Kormann_x)/np.array(pixel_size)
    Psi_Kormann_yy, Psi_Kormann_yx = np.gradient(Psi_Kormann_y)/np.array(pixel_size)

    kappa = 0.5 * (Psi_Kormann_xx + Psi_Kormann_yy)
    shear1 = 0.5 * (Psi_Kormann_xx - Psi_Kormann_yy)
    shear2 = Psi_Kormann_xy
    shear = np.sqrt(shear1**2 + shear2**2)

    miu_inversed_Kormann = (1-kappa)**2 - shear**2
    
    return miu_inversed_Kormann

def GetMag(img_position, lpar_local, pixel_iter):
    
    img_x = img_position[0]
    img_y = img_position[1]

    pixel_size_small = pixel_iter

    xmin = img_x - 10 * pixel_size_small
    xmax = img_x + 10 * pixel_size_small
    ymin = img_y - 10 * pixel_size_small
    ymax = img_y + 10 * pixel_size_small

    xrange = np.linspace(xmin, xmax, 21)
    yrange = np.linspace(ymin, ymax, 21)

    plane_xx_small, plane_yy_small = np.meshgrid(xrange, yrange)

    miu_inversed_matrix = GetInversedMiu(plane_xx_small,plane_yy_small, pixel_size_small, lpar_local)
    
    miu_inversed = miu_inversed_matrix[11][11]

    return 1./miu_inversed

def GetMagAnlytcl(theta_x, theta_y, lpar_local):
    return 1./(1-2*KappaAnlytcl(theta_x, theta_y, lpar_local))

def GetSourcePosition(caustic_outer):
    length = len(caustic_outer)
    xindx = list(range(0,length*2,2))
    yindx = list(range(1,length*2,2))
    xvalue = np.array(caustic_outer).reshape(-1,1)[xindx]
    yvalue = np.array(caustic_outer).reshape(-1,1)[yindx]
    xbound = [min(xvalue),max(xvalue)]
    ybound = [min(yvalue),max(yvalue)]
    
    while True:
        x_rand = r.uniform(xbound[0],xbound[1])
        y_rand = r.uniform(xbound[0],xbound[1])
        point = Point(x_rand, y_rand)
        polygon = Polygon(caustic_outer)
        
        if polygon.contains(point):
            break
    return (x_rand[0], y_rand[0])

def ImgScaling(ori_img):
    max_value = ori_img.max()
    min_value = ori_img.min()
    scale = max_value - min_value
    max_int16 = 2**16 - 1
    img_scaled = max_int16 * ((ori_img - min_value)/scale)
    img_scaled_int = img_scaled.astype(np.int64)
    
    return img_scaled_int

@jit(nopython=numba_flag)
def IsInside(vertex1, vertex2, vertex3, beta_ra_dec):
    # 求三个小三角形的面积
    

    s1 = GetArea(vertex1, vertex2, beta_ra_dec)
    s2 = GetArea(vertex2, vertex3, beta_ra_dec)
    s3 = GetArea(vertex1, vertex3, beta_ra_dec)
    
#     print("s1=",s1)

    if  np.around(s1 + s2 + s3,14) > np.around(GetArea(vertex1, vertex2, vertex3),14):
        return False
    else:
        return True
    
def KappaAnlytcl(theta_x, theta_y, lpar_local):
    xx,yy = XYTransform(theta_x, theta_y, lpar_local[1], lpar_local[2], lpar_local[3])
    qlens = lpar_local[4]
    theta_E = lpar_local[0]
    return 0.5*np.sqrt(qlens)*theta_E/np.sqrt(xx**2+qlens**2*yy**2)
    
def KormannPtntl(theta_x, theta_y, lpar_local):
    xx,yy = XYTransform(theta_x, theta_y, lpar_local[1], lpar_local[2], lpar_local[3])
    theta_E = lpar_local[0]
    qlens = lpar_local[4]
    qfact = np.sqrt(1-qlens**2)
    delta = np.sqrt((xx**2 + qlens**2 * yy**2)/(xx**2 + yy**2))
    Psi_Kormann = theta_E * (np.sqrt(qlens)/qfact) * \
                  (abs(yy)*np.arccos(delta) + abs(xx)*np.arccosh(delta/qlens))

    return Psi_Kormann

def PlotTrngl(tris_arr):
    for index in range(0,len(tris_arr)):
        plt.plot([tris_arr[index][0][0],
              tris_arr[index][1][0],
              tris_arr[index][2][0],
              tris_arr[index][0][0]],
        [tris_arr[index][0][1],
         tris_arr[index][1][1],
         tris_arr[index][2][1],
         tris_arr[index][0][1]])
    
    axes = plt.gca()
    axes.set_aspect(1)
    
def SplitingTriangle(tri):
    vertex1 = tri[0]
    vertex2 = tri[1]
    vertex3 = tri[2]

    vertex4 = (vertex1 + vertex2) / 2
    vertex5 = (vertex1 + vertex3) / 2
    vertex6 = (vertex2 + vertex3) / 2

    side1 = GetDistance(vertex1, vertex2)
    side2 = GetDistance(vertex1, vertex3)
    side3 = GetDistance(vertex2, vertex3)

    longest_side = max(side1, side2, side3)

    if longest_side == side1:
        vertex7 = (vertex4 + vertex1) / 2
        vertex8 = (vertex4 + vertex2) / 2
        vertex9 = (vertex5 + vertex6) / 2

        return np.array([
            [vertex1, vertex5, vertex7],
            [vertex5, vertex7, vertex9],
            [vertex4, vertex7, vertex9],
            [vertex3, vertex5, vertex9],
            [vertex4, vertex8, vertex9],
            [vertex6, vertex8, vertex9],
            [vertex3, vertex6, vertex9],
            [vertex2, vertex6, vertex8]
        ])

    elif longest_side == side2:
        vertex7 = (vertex5 + vertex1) / 2
        vertex8 = (vertex5 + vertex3) / 2
        vertex9 = (vertex4 + vertex6) / 2

        return np.array([
            [vertex1, vertex4, vertex7],
            [vertex2, vertex4, vertex9],
            [vertex4, vertex7, vertex9],
            [vertex5, vertex7, vertex9],
            [vertex2, vertex6, vertex9],
            [vertex6, vertex8, vertex9],
            [vertex5, vertex8, vertex9],
            [vertex3, vertex6, vertex8]
        ])
    else:
        vertex7 = (vertex6 + vertex2) / 2
        vertex8 = (vertex6 + vertex3) / 2
        vertex9 = (vertex4 + vertex5) / 2

        return np.array([
            [vertex1, vertex4, vertex9],
            [vertex2, vertex4, vertex7],
            [vertex4, vertex7, vertex9],
            [vertex6, vertex7, vertex9],
            [vertex1, vertex5, vertex9],
            [vertex6, vertex8, vertex9],
            [vertex5, vertex8, vertex9],
            [vertex3, vertex5, vertex8]
        ])
    
def SplitingTriangles(tris):
    tris_segmented = []
    for tri in tris:
        tris_segmented.append(SplitingTriangle(tri))

    return np.array(tris_segmented).reshape(-1, 3, 2)

def TimeDelay(img_ra_dac_sort_arr, lpar_local, zd,zs):
    if len(img_ra_dac_sort_arr) == 4:
        t_A, t_B = TwoImgTimeDelay(img_ra_dac_sort_arr, lpar_local, zd,zs)
        return t_A, t_B
    elif len(img_ra_dac_sort_arr) == 8:
        t_A, t_B, t_C, t_D = FourImgTimeDelay(img_ra_dac_sort_arr, lpar_local, zd,zs)
        return t_A, t_B, t_C, t_D
    else:
        print("Require Two/Four Imgs, but ", len(img_ra_dac_sort_arr)/2," are given.")
        return False

@jit(nopython=numba_flag)
def TriangleImg2Source(tri_img_plane, lpar_local):
    tri_source_plane = []

    for index in range(len(tri_img_plane)):
        vertex1 = tri_img_plane[index][0]
        vertex2 = tri_img_plane[index][1]
        vertex3 = tri_img_plane[index][2]

        vertex1_x_source, vertex1_y_source = DefSIE(vertex1[0], vertex1[1], lpar_local)
        vertex2_x_source, vertex2_y_source = DefSIE(vertex2[0], vertex2[1], lpar_local)
        vertex3_x_source, vertex3_y_source = DefSIE(vertex3[0], vertex3[1], lpar_local)

        tri_source_plane.append([[vertex1_x_source, vertex1_y_source],
                                 [vertex2_x_source, vertex2_y_source],
                                 [vertex3_x_source, vertex3_y_source]]
                                )

    tri_source_plane = np.array(tri_source_plane)

    return tri_source_plane

def TwoImgTimeDelay(img_ra_dac_sort_arr, lpar_local, zd,zs):
    
    if len(img_ra_dac_sort_arr)!=4:
        print("Require Two Imgs, but ", len(img_ra_dac_sort_arr)/2," are given.")
        return False
    else:
        theta_A_x = img_ra_dac_sort_arr[0]
        theta_A_y = img_ra_dac_sort_arr[1]
        theta_B_x = img_ra_dac_sort_arr[2]
        theta_B_y = img_ra_dac_sort_arr[3]

        _,Fermat_A = FermatPtntl(theta_A_x, theta_A_y, lpar_local)
        _,Fermat_B = FermatPtntl(theta_B_x, theta_B_y, lpar_local)
        deltaFermat = Fermat_A - Fermat_B

        Dd =  cosmo.angular_diameter_distance(zd).si.value               #m
        Ds =  cosmo.angular_diameter_distance(zs).si.value               #m
        Dds = cosmo.angular_diameter_distance_z1z2(zd,zs).si.value       #m

        t_AB = (1/c.day_s)*(1/c.c)*((1+zd)*(Dd*Ds)/Dds)*\
               (Fermat_A-Fermat_B)                                       #day*arcsec^2
        t_AB = t_AB * c.arcsec ** 2                                      #day
        
        t_A = 0; t_B = t_A - t_AB;
        t_earliest = min(t_A, t_B)
        t_A = t_A - t_earliest; t_B = t_B - t_earliest;
        
        return t_A,t_B

@jit(nopython=numba_flag)
def XYTransform(x, y, x_cen, y_cen, phi):

    xnew = (x - x_cen) * np.cos(np.pi * phi / 180.0) + (y - y_cen) * np.sin(np.pi * phi / 180.0)
    ynew = -(x - x_cen) * np.sin(np.pi * phi / 180.0) + (y - y_cen) * np.cos(np.pi * phi / 180.0)
    return (xnew, ynew)




