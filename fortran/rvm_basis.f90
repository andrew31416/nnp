!===================================================================!
! NOTES                                                             !
! -----                                                             !
!   - for OpenMP, compile with  --f90flags="-fopenmp" -lgomp        !
!===================================================================!

module rvm_basis
    implicit none

    contains
        real(8) function wrapped_polynomial(k,r_s,x)
            implicit none

            real(8),intent(in) :: k,r_s,x

            wrapped_polynomial = (x/r_s)**k
        end function wrapped_polynomial

        real(8) function wrapped_cos(k,r_s,kspacing,r)
            implicit none

            real(8),intent(in) :: k,r_s,r,kspacing

            wrapped_cos = cos(3.14159265359d0*r*k*kspacing/r_s)
        end function wrapped_cos

        real(8) function wrapped_cos2(k,kspacing,theta)
            implicit none

            real(8),intent(in) :: k,theta,kspacing
        
            wrapped_cos2 = cos(k*theta*kspacing)
        end function wrapped_cos2

        real(8) function wrapped_smooth(f_s,r_s,cuthigh,r)
            implicit none

            real(8),intent(in) :: f_s,r_s,r
            logical,intent(in) :: cuthigh

            real(8) :: x,tmp

            x = (r-r_s)/f_s

            if (cuthigh .and. x.gt.0.0d0) then
                x = 0.0d0
            else if ((cuthigh.eqv..false.) .and. (x.lt.0.0d0)) then
                x = 0.0d0
            end if
        
            tmp = x**4

            wrapped_smooth = tmp/(1.0d0+tmp)
        end function wrapped_smooth

        real(8) function wrapped_guassian(k,r_s,kspacing,r)
            implicit none

            real(8),intent(in) :: k,r_s,kspacing,r

            wrapped_guassian = exp(-(kspacing*k*r/r_s)**2)
        end function wrapped_guassian

        real(8) function basis_wrapper(k,r_s,kspacing,r,basis_type)
            !===================================================================!
            ! wrapper to provide interface between phi matrix generator and     !
            ! specific basis functions                                          !
            !===================================================================!
            ! Input                                                             !
            ! -----                                                             !
            !   - k          : k in N,[0,M-1]                                   !
            !   - r_s        : cut off distance of interaction                  !
            !   - kspacing   : interval between successive points in kspace     !
            !   - r          : distance                                         !
            !   - basis_type : an integer interfacing basis function type to    !
            !                  functions                                        !
            !                                                                   !
            ! |==============================================================!  !
            ! |basis_type | function_name      | function                    !  !
            ! |------------------------------------- ------------------------!  !
            ! |0          | wrapped_cos        | cos(pi*k*kspacing*r/r_s)    !  !
            ! |1          | wrapped_polynomial | (r/r_s)**k                  !  !
            ! |2          | wrapped_guassian   | exp(-[r/r_s*k*kspacing]**2) !  !
            ! !==============================================================!  !
            !                                                                   !
            !===================================================================!   
            ! Output                                                            !
            ! ------                                                            !
            !   - function_name : projection in a given 1-d basis function      !
            !===================================================================!

            implicit none

            !* arguments
            real(8),intent(in) :: k,r_s,kspacing,r
            integer,intent(in) :: basis_type

            !* scratch
            real(8) :: tmp = 0.0d0

            if (basis_type.eq.0) then 
                tmp = wrapped_cos(k,r_s,kspacing,r)
            else if (basis_type.eq.1) then
                tmp = wrapped_polynomial(k,r_s,r)
            else if (basis_type.eq.2) then
                tmp = wrapped_guassian(k,r_s,kspacing,r)
            else
                !* better to exit than to stay quiet
                call exit_program("basis_wrapper")
            end if

            basis_wrapper = tmp
        end function basis_wrapper

        subroutine isotropic_phi_cos(rvalues,kvals,f_s,r_s,smooth,kspacing_iso,Nk,Nbonds,Nmaxr,Nr,&
            &basis_type,phi)
            !===================================================================!
            ! calculate isotropic segment of phi matrix (i,j) for ith bond, kth !
            ! k value where i=[1,Nbonds] , j=[1,Nk]                             !
            !===================================================================!
            ! Input                                                             !
            ! -----                                                             !
            !   - rvalues : a (Nmaxr,Nbonds) array where rvalues(i,j) is the    !
            !               ith r value of the jth bond for the given species   !
            !   - kvals   : a (Nk) array of k values                            !
            !   - f_s     : scaling factor for k values                         !
            !   - r_s     : array for smoothing parameter (cut off radii)       !
            !               if r_s(2) == -1 then only taper at r_s(1),          !
            !               otherwise taper all contributions below r_s(2) too  !
            !   - smooth  : whether or not to include smoothing as r-> 0        !
            !   - Nk      : number of functions in fourier set                  !
            !   - Nbonds  : number of bonds                                     !
            !   - Nmaxr   : max length of r arrays over all considered bonds    !
            !   - Nr      : (Nbonds) array giving the number of r values        !
            !                present in each bond                               !
            !   - basis_type : key specifying basis function type               !
            !-------------------------------------------------------------------!
            ! Output                                                            !
            ! ------                                                            !         
            !   - phi     : (Nbonds,Nk) array for phi matrix. phi(i,j) is the   !
            !               contribution from the ith bond and jth fourier      !
            !               function                                            !
            !===================================================================!

            
            implicit none

            integer,intent(in) :: Nk,Nbonds,Nmaxr,Nr(1:Nbonds),basis_type
            real(8),intent(in) :: rvalues(1:Nmaxr,1:Nbonds),kvals(1:Nk),f_s,r_s(1:2),kspacing_iso
            logical,intent(in) :: smooth
            real(8),intent(out) :: phi(1:Nbonds,1:Nk)

            !* scratch
            integer :: ii,jj,kk
            real(8) :: tmp,r,k

            do ii=1,Nk,1
                !* loop over k

                k = kvals(ii)

                do jj=1,Nbonds,1
                    !* loop over bonds

                    tmp = 0.0d0

                    do kk=1,Nr(jj),1
                        !* [r][bond]
                        r = rvalues(kk,jj)

                        if (smooth) then
                            if (r_s(2).lt.-0.0000001d0) then
                                ! do not taper contribution at r_s(2)
                                tmp = tmp + basis_wrapper(k,r_s(1),kspacing_iso,r,basis_type)*&
                                    &*wrapped_smooth(f_s,r_s(1),.true.,r)
                            else
                                !* taper contribution at r_s(2) < r_s(1) too
                                tmp = tmp + basis_wrapper(k,r_s(1),kspacing_iso,r,basis_type)*&
                                    &wrapped_smooth(f_s,r_s(1),.true.,r)*&
                                    &wrapped_smooth(f_s,r_s(2),.false.,r)
                            end if
                        else
                            tmp = tmp + basis_wrapper(k,r_s(1),kspacing_iso,r,basis_type)
                        end if
                    end do

                    !* [bond][k]
                    phi(jj,ii) = tmp
                end do
            end do

        end subroutine

        subroutine anisotropic_phi_cos(r_thetas,kvals,f_s,r_s,smooth,kspacing,Nk,Nbonds,Neighmax,&
                &Nr,function_type,phi)
            !=============================================================!
            !                                                             !
            ! ----------------------------------------------------------  !
            ! function_type |
            ! ----------------------------------------------------------
            !       0       | cos(k1*r1/r_s)*cos(k2*r2/r_s)*cos(k3*theta) !
            !       1       | (r1**k1)(r2**k2)(theta**k3)                 !
            ! ----------------------------------------------------------- !
            !                                                             !    
            !=============================================================!
            
            implicit none

            !* arguments
            integer,intent(in) :: Nk,Nbonds,Neighmax,Nr(1:Nbonds),function_type
            real(8),intent(in) :: r_thetas(1:3,1:Neighmax,1:Nbonds),kvals(1:3,1:Nk)
            real(8),intent(in) :: f_s(1:2),r_s,kspacing(1:2)
            logical,intent(in) :: smooth
            real(8),intent(out) :: phi(1:Nbonds,1:Nk)

            !* scratch
            integer :: ii,jj,kk
            real(8) :: k1,k2,k3,tmp,dr1,dr2,dtheta12

            do ii=1,Nk,1
                k1 = kvals(1,ii)
                k2 = kvals(2,ii)
                k3 = kvals(3,ii)

                do jj=1,Nbonds,1
                    !* loop over bonds

                    tmp = 0.0d0

                    do kk=1,Nr(jj),1
                        !* loop over neighbour entries for bond

                        dr1      = r_thetas(1,kk,jj)
                        dr2      = r_thetas(2,kk,jj)
                        dtheta12 = r_thetas(3,kk,jj)
                        
                        if (function_type.eq.0) then
                            !* cosine product
                            if (smooth) then
                                tmp = tmp + wrapped_cos(k1,r_s,kspacing(1),dr1)*wrapped_cos(k2,r_s,kspacing(1),dr2)*&
                                        &wrapped_cos2(k3,kspacing(2),dtheta12)*&
                                        &wrapped_smooth(f_s(1),r_s,.true.,dr1)*&
                                        &wrapped_smooth(f_s(1),r_s,.true.,dr2)*&
                                        &wrapped_smooth(f_s(2),0.0d0,.false.,dr1)*&
                                        &wrapped_smooth(f_s(2),0.0d0,.false.,dr2)
                            else 
                                tmp = tmp + wrapped_cos(k1,r_s,kspacing(1),dr1)*wrapped_cos(k2,r_s,kspacing(1),dr2)*&
                                wrapped_cos2(k3,kspacing(2),dtheta12)
                            end if
                        else if (function_type.eq.1) then
                            !* multivariate polynomail basis set
                            if (smooth) then
                                !* taper at rcut and 0 for dr terms
                                tmp = tmp + wrapped_polynomial(k1,r_s,dr1)*wrapped_polynomial(k2,r_s,dr2)*&
                                        &wrapped_polynomial(k3,3.14159265359d0,dtheta12)*&
                                        &wrapped_smooth(f_s(1),r_s,.true.,dr1)*&
                                        &wrapped_smooth(f_s(1),r_s,.true.,dr2)*&
                                        &wrapped_smooth(f_s(2),0.0d0,.false.,dr1)*&
                                        &wrapped_smooth(f_s(2),0.0d0,.false.,dr2)
                            else 
                                tmp = tmp + wrapped_polynomial(k1,r_s,dr1)*wrapped_polynomial(k2,r_s,dr2)*&
                                        &wrapped_polynomial(k3,3.14159265359d0,dtheta12)
                            end if

                        end if
                    
                    end do
                    
                    phi(jj,ii) = tmp
                
                end do
            end do

        end subroutine anisotropic_phi_cos

        subroutine openmp_isotropic_phi_cos(rvalues,kvals,f_s,r_s,smooth,kspacing_iso,Nk,Nbonds,Nmaxr,Nr,&
            &basis_type,phi)
            !===================================================================!
            ! OpenMP multi-threaded version of isotropic_phi_cos                !
            !                                                                   !
            ! NOTE                                                              !
            ! ----                                                              !
            !   - loop over bonds is parallelised here as this will tend to inf !
            !     not the number of basis functions k, as training set tends to !
            !     inf                                                           !
            !                                                                   !
            !===================================================================!
            ! Input                                                             !
            ! -----                                                             !
            !   - rvalues : a (Nmaxr,Nbonds) array where rvalues(i,j) is the    !
            !               ith r value of the jth bond for the given species   !
            !   - kvals   : a (Nk) array of k values                            !
            !   - f_s     : scaling factor for k values                         !
            !   - r_s     : 2 value array with r_s(1) = larger cut off radius   !
            !               r_s(2) = smaller taper radius. If r_s(2)==-1 then   !
            !               no lower tapering is executed                       !
            !   - smooth  : whether or not to include smoothing as r-> 0        !
            !   - Nk      : number of functions in fourier set                  !
            !   - Nbonds  : number of bonds                                     !
            !   - Nmaxr   : max length of r arrays over all considered bonds    !
            !   - Nr      : (Nbonds) array giving the number of r values        !
            !                present in each bond                               !
            !   - basis_type : key specifying basis function type               !
            !-------------------------------------------------------------------!
            ! Output                                                            !
            ! ------                                                            !         
            !   - phi     : (Nbonds,Nk) array for phi matrix. phi(i,j) is the   !
            !               contribution from the ith bond and jth fourier      !
            !               function                                            !
            !===================================================================!

            !$ use omp_lib
            
            implicit none

            integer,intent(in) :: Nk,Nbonds,Nmaxr,Nr(1:Nbonds),basis_type
            real(8),intent(in) :: rvalues(1:Nmaxr,1:Nbonds),kvals(1:Nk),f_s,r_s(1:2),kspacing_iso
            logical,intent(in) :: smooth
            real(8),intent(out) :: phi(1:Nbonds,1:Nk)

            !* scratch
            integer :: ii,jj,kk,thread,num_threads,start_idx,final_idx,dj
            real(8) :: tmp,r,k
            
            do ii=1,Nk,1
                !* loop over k

                k = kvals(ii)
                
                !$omp parallel num_threads(omp_get_max_threads()),&
                !$omp private(tmp,r,thread,num_threads,dj,start_idx,final_idx,jj,kk),&
                !$omp shared(k,rvalues,phi,ii,Nr,kvals,smooth,r_s,f_s)

                !* thread idx = [0,num_threads]
                thread = omp_get_thread_num()

                !* number of threads opened
                num_threads = omp_get_max_threads()

                !*--------------*!
                !* balance load *!
                !*--------------*!
    
                !* number of jj per thread expcept for final thread
                dj = int(floor(float(Nbonds)/float(num_threads)))
              
                !* always true
                start_idx = thread*dj + 1

                if (thread.eq.num_threads-1) then
                    final_idx = Nbonds
                else    
                    final_idx = (thread+1)*dj
                end if


                do jj=start_idx,final_idx,1
                    !* loop over bonds -> this tends to inf 

                    tmp = 0.0d0

                    do kk=1,Nr(jj),1
                        !* [r][bond]
                        r = rvalues(kk,jj)

                        if (smooth) then
                            if (r_s(2).lt.-0.0000001d0) then
                                ! do not taper contribution at r_s(2)
                                tmp = tmp + basis_wrapper(k,r_s(1),kspacing_iso,r,basis_type)*&
                                    &wrapped_smooth(f_s,r_s(1),.true.,r)
                            else
                                !* taper contribution at r_s(2) < r_s(1) too
                                tmp = tmp + basis_wrapper(k,r_s(1),kspacing_iso,r,basis_type)*&
                                    &wrapped_smooth(f_s,r_s(1),.true.,r)*&
                                    &wrapped_smooth(f_s,r_s(2),.false.,r)
                            end if
                        else
                            tmp = tmp + basis_wrapper(k,r_s(1),kspacing_iso,r,basis_type)
                        end if
                    end do

                    !* [bond][k]
                    phi(jj,ii) = tmp
                end do

                !$omp end parallel
            end do

        end subroutine openmp_isotropic_phi_cos
        
        subroutine openmp_anisotropic_phi_cos(r_thetas,kvals,f_s,r_s,smooth,kspacing,Nk,Nbonds,Neighmax,&
                &Nr,function_type,phi)
            !$ use omp_lib
            
            implicit none

            !* arguments
            integer,intent(in) :: Nk,Nbonds,Neighmax,Nr(1:Nbonds),function_type
            real(8),intent(in) :: r_thetas(1:3,1:Neighmax,1:Nbonds),kvals(1:3,1:Nk)
            real(8),intent(in) :: f_s(1:2),r_s,kspacing(1:2)
            logical,intent(in) :: smooth
            real(8),intent(out) :: phi(1:Nbonds,1:Nk)

            !* scratch
            integer :: ii,jj,kk,dj,start_idx,final_idx,thread,num_threads
            real(8) :: k1,k2,k3,tmp,dr1,dr2,dtheta12

            do ii=1,Nk,1
                k1 = kvals(1,ii)
                k2 = kvals(2,ii)
                k3 = kvals(3,ii)
                
                !$omp parallel num_threads(omp_get_max_threads()),&
                !$omp private(thread,num_threads,dj,start_idx,final_idx,jj,kk,dr1,dr2,dtheta12,tmp),&
                !$omp shared(k1,k2,k3,ii,Nr,r_thetas,r_s,f_s,smooth,Nbonds,phi,kvals)
                
                !* thread idx = [0,num_threads]
                thread = omp_get_thread_num()

                !* number of threads opened
                num_threads = omp_get_max_threads()

                !*--------------*!
                !* balance load *!
                !*--------------*!
    
                !* number of jj per thread expcept for final thread
                dj = int(floor(float(Nbonds)/float(num_threads)))
              
                !* always true
                start_idx = thread*dj + 1

                if (thread.eq.num_threads-1) then
                    final_idx = Nbonds
                else    
                    final_idx = (thread+1)*dj
                end if

                do jj=start_idx,final_idx,1
                    !* loop over bonds

                    tmp = 0.0d0

                    do kk=1,Nr(jj),1
                        !* loop over neighbour entries for bond

                        dr1      = r_thetas(1,kk,jj)
                        dr2      = r_thetas(2,kk,jj)
                        dtheta12 = r_thetas(3,kk,jj)

                        if (function_type.eq.0) then
                            !* cosine product series
                            if (smooth) then
                                tmp = tmp + wrapped_cos(k1,r_s,kspacing(1),dr1)*wrapped_cos(k2,r_s,kspacing(1),dr2)*&
                                        &wrapped_cos2(k3,kspacing(2),dtheta12)*&
                                        &wrapped_smooth(f_s(1),r_s,.true.,dr1)*&
                                        &wrapped_smooth(f_s(1),r_s,.true.,dr2)*&
                                        &wrapped_smooth(f_s(2),0.0d0,.false.,dr1)*&
                                        &wrapped_smooth(f_s(2),0.0d0,.false.,dr2)
                            else 
                                tmp = tmp + wrapped_cos(k1,r_s,kspacing(1),dr1)*wrapped_cos(k2,r_s,kspacing(1),dr2)*&
                                        wrapped_cos2(k3,kspacing(2),dtheta12)
                            end if
                        else if (function_type.eq.1) then
                            !* multivariate polynomial basis
                            if (smooth) then
                                tmp = tmp + wrapped_polynomial(k1,r_s,dr1)*wrapped_polynomial(k2,r_s,dr2)*&
                                        &wrapped_polynomial(k3,3.14159265359d0,dtheta12)*&
                                        &wrapped_smooth(f_s(1),r_s,.true.,dr1)*&
                                        &wrapped_smooth(f_s(1),r_s,.true.,dr2)*&
                                        &wrapped_smooth(f_s(2),0.0d0,.false.,dr1)*&
                                        &wrapped_smooth(f_s(2),0.0d0,.false.,dr2)
                            else
                                tmp = tmp + wrapped_polynomial(k1,r_s,dr1)*wrapped_polynomial(k2,r_s,dr2)*&
                                        &wrapped_polynomial(k3,3.14159265359d0,dtheta12)
                            end if
                        end if

                    end do
                    
                    phi(jj,ii) = tmp
                
                end do
                
                !$omp end parallel
            end do

        end subroutine openmp_anisotropic_phi_cos

        subroutine exit_program(function_name)
            implicit none

            character(len=*),intent(in) :: function_name
    
            write(*,*) "\n-----------------------------"
            write(*,*) "Error raised in rvm_basis.f90"
            write(*,*) "-----------------------------"
            write(*,*) "Function : ",function_name
            call flush()
            call exit(0)
        end subroutine exit_program
end module
