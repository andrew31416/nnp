module propagate
    use config
    use io
    use feature_util, only : int_in_intarray
    use util, only : allocate_dydx, zero_weights, scalar_equal
    use init, only : allocate_weights
    use omp_lib

    implicit none

    external :: dgemv
    external :: dgemm
    external :: dcopy
    real(8),external :: ddot
            
    type(weights),public,save :: d2ydxdw
    real(8),public,allocatable,save :: sub_A1(:,:),sub_A2(:,:),sub_A2A1(:,:)
    real(8),public,allocatable,save :: sub_B(:,:),sub_C(:,:),sub_D(:,:),sub_BT(:,:)
    real(8),public,allocatable,save :: hprimeprime_1(:),hprimeprime_2(:),tmp_l1(:)
    real(8),public,allocatable,save :: d2ydx2(:,:)

    !$omp threadprivate(d2ydxdw)
    !$omp threadprivate(sub_A1)
    !$omp threadprivate(sub_A2)
    !$omp threadprivate(sub_A2A1)
    !$omp threadprivate(sub_B)
    !$omp threadprivate(sub_C)
    !$omp threadprivate(sub_D)
    !$omp threadprivate(sub_BT)
    !$omp threadprivate(hprimeprime_1)
    !$omp threadprivate(hprimeprime_2)
    !$omp threadprivate(tmp_l1)
    !$omp threadprivate(d2ydx2)


    contains
        subroutine forward_propagate(set_type,conf)
            implicit none

            integer,intent(in) :: conf,set_type

            !* scratch
            integer :: ii,natm,atm

            natm = data_sets(set_type)%configs(conf)%n

            !------------------!
            !* hidden layer 1 *!
            !------------------!
!write(*,*) ''
!write(*,*) 'x:'
!write(*,*) data_sets(set_type)%configs(conf)%x(3,:)            
            
            !* a^1_mI = \sum_k=0^K w_mk^1 gamma_kI
            call dgemm('n','n',net_dim%hl1,natm,D+1,1.0d0,net_weights%hl1,net_dim%hl1,&
                    &data_sets(set_type)%configs(conf)%x,D+1,0.0d0,net_units%a%hl1,net_dim%hl1)

            do atm=1,natm,1
                !* null value for bias
                net_units%z%hl1(0,atm) = 1.0d0

                do ii=1,net_dim%hl1,1
                    !* non-linear activation
                    net_units%z%hl1(ii,atm) = activation(net_units%a%hl1(ii,atm))
                end do
            end do
!write(*,*) 'first layer output:',net_units%z%hl1(:,1)            

            !------------------!
            !* hidden layer 2 *!
            !------------------!

            !* a^2_lI = \sum_{m=0}^N1 w^2_lm z^1_mI
            call dgemm('n','n',net_dim%hl2,natm,net_dim%hl1+1,1.0d0,net_weights%hl2,net_dim%hl2,&
                    &net_units%z%hl1,net_dim%hl1+1,0.0d0,net_units%a%hl2,net_dim%hl2)

            do atm=1,natm,1
                !* null value for bias
                net_units%z%hl2(0,atm) = 1.0d0

                do ii=1,net_dim%hl2,1
                    !* nonlinear map of linear models
                    net_units%z%hl2(ii,atm) = activation(net_units%a%hl2(ii,atm))
                end do
            end do
!write(*,*) 'second layer output:',net_units%z%hl1(:,1)            

            !-------------------------!
            !* final layer to output *!
            !-------------------------!
           
            !* try to convert this to dgemv with 't'  
            do atm=1,natm,1
                data_sets(set_type)%configs(conf)%current_ei(atm) = ddot(net_dim%hl2+1,&
                        &net_weights%hl3,1,net_units%z%hl2(:,atm),1)
            end do
!write(*,*) 'output:',data_sets(set_type)%configs(conf)%current_ei(1)            
        end subroutine

        subroutine backward_propagate(set_type,conf)
            implicit none

            integer,intent(in) :: conf,set_type

            !* scratch
            integer :: ii,atm,natm
    
            natm = data_sets(set_type)%configs(conf)%n
            
            !-------------------!
            !* delta back prop *!
            !-------------------!

            !* layer 2 *!
           
            do atm=1,natm,1 
                !* delta_i^(2) = h'(a_i^(2)) * w_i^(3) 
                do ii=1,net_dim%hl2,1
                    !* activation derivatives
                    net_units%a_deriv%hl2(ii,atm) = activation_deriv(net_units%a%hl2(ii,atm))
                    
                    net_units%delta%hl2(ii,atm) = net_units%a_deriv%hl2(ii,atm)*net_weights%hl3(ii)
                end do
            end do
            
            !* layer 1 *!
            
            !* delta_iI^(1) = h'(a_i^(1)) * sum_j w_ji^(2) delta_jI^(2)
            call dgemm('n','n',net_dim%hl1,natm,net_dim%hl2,1.0d0,net_weights_nobiasT%hl2,&
                    &net_dim%hl1,net_units%delta%hl2,net_dim%hl2,0.0d0,net_units%delta%hl1,net_dim%hl1)
           
            do atm=1,natm,1 
                do ii=1,net_dim%hl1,1
                    !* activation derivatives
                    net_units%a_deriv%hl1(ii,atm) = activation_deriv(net_units%a%hl1(ii,atm))

                    net_units%delta%hl1(ii,atm) = net_units%a_deriv%hl1(ii,atm)*net_units%delta%hl1(ii,atm)
                end do
            end do
            
            !-----------------------------------------!
            !* derivative of net output wrt features *!
            !-----------------------------------------!
   
            call dgemm('n','n',D,natm,net_dim%hl1,1.0d0,net_weights_nobiasT%hl1,D,&
                    &net_units%delta%hl1,net_dim%hl1,0.0d0,dydx,D)

        end subroutine backward_propagate

        subroutine calculate_dydw(set_type,conf,atm)
            implicit none

            !* args
            integer,intent(in) :: set_type,conf,atm

            !---------------!
            !* third layer *!
            !---------------!
            
            !* dydw%hl3 = z%hl2 , shape=(N2+1,)
            call dcopy(net_dim%hl2+1,net_units%z%hl2(:,atm),1,dydw%hl3,1)
            
            !----------------!
            !* second layer *!
            !----------------!
         
            ! dydw%hl2_lm =  delta%hl2_l  *  z%hl1_m   l=[1,N2] , m=[0,N1]
            call dgemm('n','n',net_dim%hl2,net_dim%hl1+1,1,1.0d0,net_units%delta%hl2(:,atm),net_dim%hl2,&
                    &net_units%z%hl1(:,atm),1,0.0d0,dydw%hl2,net_dim%hl2)
            
            !---------------!
            !* first layer *!
            !---------------!

            call dgemm('n','n',net_dim%hl1,D+1,1,1.0d0,net_units%delta%hl1(:,atm),net_dim%hl1,&
                    &data_sets(set_type)%configs(conf)%x(:,atm),1,0.0d0,dydw%hl1,net_dim%hl1)
        end subroutine calculate_dydw

        subroutine init_forceloss_subsidiary_mem()
            implicit none

            allocate(sub_A1(net_dim%hl1,D))
            allocate(sub_A2(net_dim%hl2,net_dim%hl1))
            allocate(sub_A2A1(net_dim%hl2,D))
            allocate(sub_B(net_dim%hl2,D))
            allocate(sub_BT(1:D,1:net_dim%hl2))
            allocate(sub_C(net_dim%hl2,net_dim%hl1))
            allocate(sub_D(D,net_dim%hl1))
            allocate(hprimeprime_2(1:net_dim%hl2))
            allocate(hprimeprime_1(1:net_dim%hl1))
            allocate(tmp_l1(1:net_dim%hl1))
        end subroutine init_forceloss_subsidiary_mem

        subroutine deallocate_forceloss_subsidiary_mem()
            implicit none
            deallocate(sub_A1)
            deallocate(sub_A2)
            deallocate(sub_A2A1)
            deallocate(sub_B)
            deallocate(sub_BT)
            deallocate(sub_C)
            deallocate(sub_D)
            deallocate(hprimeprime_1)
            deallocate(hprimeprime_2)
            deallocate(tmp_l1)
        end subroutine deallocate_forceloss_subsidiary_mem

        subroutine forceloss_weight_derivative_subsidiary1(atm)
            implicit none

            integer,intent(in) :: atm
            !* compute matrices A_1,A_2,B,C for conf,atm
    
            integer :: ii,jj
            
            do ii=1,net_dim%hl1
                do jj=1,net_dim%hl2
                    !* A2_ji = h'(a^2_j)*w^2_ji
                    sub_A2(jj,ii) = net_units%a_deriv%hl2(jj,atm)*net_weights%hl2(jj,ii)
                end do
            end do

            do ii=1,D
                do jj=1,net_dim%hl1
                    !* A1_ji = h'(a^1_j)*w^1_ji
                    sub_A1(jj,ii) = net_units%a_deriv%hl1(jj,atm)*net_weights%hl1(jj,ii)
                end do
            end do

            !* A2A1_ij = \sum_k A2_{ik} A1_{kj} - Have checked
            call dgemm('n','n',net_dim%hl2,D,net_dim%hl1,1.0d0,sub_A2,net_dim%hl2,sub_A1,net_dim%hl1,&
                    &0.0d0,sub_A2A1,net_dim%hl2)


            !* B_ij = \sum_k w^2_ik A^1_kj - Have checked
            call dgemm('n','n',net_dim%hl2,D,net_dim%hl1,1.0d0,net_weights%hl2(:,1:),net_dim%hl2,sub_A1,&
                    &net_dim%hl1,0.0d0,sub_B,net_dim%hl2)
            
            
            !* second derivatives of activation
            do ii=1,net_dim%hl1
                !* h''(a^(1)))
                hprimeprime_1(ii) = activation_derivderiv(net_units%a%hl1(ii,atm)) 

                if (scalar_equal(hprimeprime_1(ii),0.0d0,dble(1e-10),dble(1e-10)**2,.false.)) then
                    !* avoid 1/0
                    tmp_l1(ii) = 0.0d0
                else
                    tmp_l1(ii) = hprimeprime_1(ii)/net_units%a_deriv%hl1(ii,atm) * net_units%delta%hl1(ii,atm)
                end if
            end do
            do ii=1,net_dim%hl2
                !* h''(a^(2))
                hprimeprime_2(ii) = activation_derivderiv(net_units%a%hl2(ii,atm))
            end do
            
            
            !* C_lp =  w^(3)_l * h''(a^(2)_l) * w^(2)_lp * h'(a^(1)_p)
            do ii=1,net_dim%hl1
                do jj=1,net_dim%hl2
                    sub_C(jj,ii) = net_weights%hl3(jj)*hprimeprime_2(jj)*net_weights%hl2(jj,ii)*&
                            &net_units%a_deriv%hl1(ii,atm)
                end do
            end do

            !* for some reason can't work dgemm with 't' , must transpose by hand
            do ii=1,D
                do jj=1,net_dim%hl2
                    sub_BT(ii,jj) = sub_B(jj,ii)
                end do
            end do
            !* D_km = sum_l^N2 B^T_kl * C_lm - Have checked
            call dgemm('n','n',D,net_dim%hl1,net_dim%hl2,1.0d0,sub_BT,D,sub_C,net_dim%hl2,0.0d0,sub_D,D)

        end subroutine forceloss_weight_derivative_subsidiary1


        subroutine forceloss_weight_derivative_subsidiary2(set_type,conf,atm,kk)
            implicit none

            integer,intent(in) :: set_type,conf,atm,kk

            integer :: ii,jj,ll,mm
            real(8) :: tmp1
            
            ! d^2 y / dwdx = d2ydxdw(atm,feature)%hl*
        
            
            !===========!
            !* layer 3 *!
            !===========!
        
            do ii=1,net_dim%hl2
                d2ydxdw%hl3(ii) = sub_A2A1(ii,kk)
            end do
            d2ydxdw%hl3(0) = 0.0d0

            !===========!
            !* layer 2 *!
            !===========!
        
            do mm=0,net_dim%hl1
                do ll=1,net_dim%hl2
                    if ( (mm.eq.0) ) then
                        !* bias
                        tmp1 = 0.0d0
                    else
                        tmp1 = sub_A1(mm,kk)*net_units%delta%hl2(ll,atm)
                    end if

                    d2ydxdw%hl2(ll,mm) = net_weights%hl3(ll)*hprimeprime_2(ll)*&
                            &net_units%z%hl1(mm,atm)*sub_B(ll,kk) + tmp1
                end do !* end loop over jj
            end do !* end loop over ii

            !===========!
            !* layer 1 *!
            !===========!

            do ii=0,D
                if (ii.eq.kk) then
                    !* bias
                    tmp1 = 1.0d0
                else
                    tmp1 = 0.0d0
                end if

                do jj=1,net_dim%hl1

                    d2ydxdw%hl1(jj,ii) = data_sets(set_type)%configs(conf)%x(ii+1,atm)*sub_D(kk,jj) +&
                            &tmp_l1(jj)*data_sets(set_type)%configs(conf)%x(ii+1,atm)*net_weights%hl1(jj,kk)+&
                            &tmp1*net_units%delta%hl1(jj,atm)

                end do !* end loop jj over 1st layer index
            end do !* end loop ii over feature index
        end subroutine forceloss_weight_derivative_subsidiary2

        subroutine calculate_forces(set_type,conf)
            implicit none

            !* args
            integer,intent(in) :: set_type,conf

            !* scratch
            integer :: atm,natm,ii,jj,deriv_idx
           
            natm = data_sets(set_type)%configs(conf)%n

            data_sets(set_type)%configs(conf)%current_fi = 0.0d0
    
            do atm=1,natm,1
                do ii=1,D,1
                    if (data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%n.eq.0) then
                        !* feature doest not contain any position info
                        cycle
                    end if
                    
                    do deriv_idx=1,data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%n,1
                        !* d feature_{ii,atm} / d r_jj
                        jj = data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%idx(deriv_idx)
                        
                        !* -= d E_atm / d feature_{ii,atm} * d feature_{ii,atm} / d r_jj
                        data_sets(set_type)%configs(conf)%current_fi(:,jj) = &
                                &data_sets(set_type)%configs(conf)%current_fi(:,jj) - dydx(ii,atm) * &
                                &data_sets(set_type)%configs(conf)%x_deriv(ii,atm)%vec(:,deriv_idx)
                    end do !* end loop over atoms jj contributing to feature (ii,atm)
                
                end do !* end loop over features ii
            end do !* end loop over local cell atoms
        end subroutine calculate_forces

!        subroutine backprop_all_forces(set_type)
!            !===================================================!
!            ! Assumes that net has been forward propagated on   !
!            ! all configurations in set_type                    !
!            !                                                   !
!            ! Parameters                                        !
!            ! ----------                                        !
!            ! set_type : int, allowed values = 1,2              !
!            !     The data set to compute forces for            !  
!            !===================================================!
!            
!            implicit none
!
!            !* args
!            integer,intent(in) :: set_type
!
!            !* scratch
!            integer :: atm,conf,ii
!            type(weights),allocatable :: d2ydxdw(:,:)
!        
!            call init_forceloss_subsidiary_mem()
!            call forceloss_weight_derivative_subsidiary()
!            
!            do conf=1,data_sets(set_type)%nconf,1
!                if (allocated(dydx)) then
!                    deallocate(dydx)
!                end if
!                call allocate_dydx(set_type,conf)
!
!                if (allocated(d2ydxdw)) then
!                    deallocate(d2ydxdw)
!                end if
!                allocate(d2ydxdw(data_sets(set_type)%configs(conf)%n,D)) 
!                do ii=1,D
!                    do atm=1,data_sets(set_type)%configs(conf)%n
!                        call allocate_weights(d2ydxdw(atm,ii))
!                        call zero_weights(d2ydxdw(atm,ii))
!                    end do
!                end do
!
!                do atm=1,data_sets(set_type)%configs(conf)%n,1
!                    call backward_propagate(conf,atm,set_type)
!                end do 
!                call calculate_forces(set_type,conf)
!
!                deallocate(dydx)
!            end do
!        end subroutine backprop_all_forces

        subroutine calculate_d2ydx2(set_type,conf)
            !* for use with feature selection
            implicit none

            !* args
            integer,intent(in) :: set_type,conf

            !* scratch
            integer :: ii,atm,ft,ll,mm
            real(8) :: hprimeprime_1(1:net_dim%hl1,1:data_sets(set_type)%configs(conf)%n)
            real(8) :: hprimeprime_2(1:net_dim%hl2,1:data_sets(set_type)%configs(conf)%n)
            real(8) :: tmp_ll(1:2)
            
            ! d2yx2(ft,atm)

            if (allocated(d2ydx2)) then
                deallocate(d2ydx2)
            end if
            allocate(d2ydx2(feature_params%num_features,data_sets(set_type)%configs(conf)%n))
            d2ydx2 = 0.0d0

            !* activation second derivatives
            do atm=1,data_sets(set_type)%configs(conf)%n,1
                do ii=1,net_dim%hl1
                    !* h''(a^(1)))
                    hprimeprime_1(ii,atm) = activation_derivderiv(net_units%a%hl1(ii,atm)) 
                end do
                do ii=1,net_dim%hl2
                    !* h''(a^(2))
                    hprimeprime_2(ii,atm) = activation_derivderiv(net_units%a%hl2(ii,atm))
                end do
            end do !* end loop over atoms

            do atm=1,data_sets(set_type)%configs(conf)%n,1
                do ft=1,feature_params%num_features
                    do ll=1,net_dim%hl2
                        tmp_ll = 0.0d0
                        do mm=1,net_dim%hl1
                            tmp_ll(1) = tmp_ll(1) + net_weights%hl2(ll,mm)*&
                                    &net_units%a_deriv%hl1(mm,atm)*net_weights%hl1(mm,ft)

                            tmp_ll(2) = tmp_ll(2) + net_weights%hl2(ll,mm)*hprimeprime_1(mm,atm)*&
                                    &(net_weights%hl1(mm,ft)**2)
                        end do

                        d2ydx2(ft,atm) = d2ydx2(ft,atm) + net_weights%hl3(ll)*&
                                &( hprimeprime_2(ll,atm)*(tmp_ll(1)**2) + &
                                &net_units%a_deriv%hl2(ll,atm)*tmp_ll(2) )
                    end do
                end do !* end loop over features
            end do !* end loop over atoms

        end subroutine calculate_d2ydx2

        real(8) function activation(ain)
            implicit none

            real(8),intent(in) :: ain

            if (nlf.eq.1) then
                activation = logistic(ain)
            else if (nlf.eq.2) then
                activation = tanh(ain)
            else
                call error("activation","unsupported nonlinear function")
                call exit(0)
            end if
        end function activation

        real(8) function activation_deriv(ain)
            implicit none

            real(8),intent(in) :: ain

            if (nlf.eq.1) then
                activation_deriv = logistic_deriv(ain)
            else if (nlf.eq.2) then
                activation_deriv = tanh_deriv(ain)
            else
                call error("activation_deriv","unsupported nonlinear function")
                call exit(0)
            end if
        end function activation_deriv

        real(8) function activation_derivderiv(ain)
            implicit none

            real(8),intent(in) :: ain

            if (nlf.eq.1) then
                activation_derivderiv = logistic_derivderiv(ain)
            else if (nlf.eq.2) then
                activation_derivderiv = tanh_derivderiv(ain)
            else
                call error("activation_derivderiv","unsupported nonlinear function")
                call exit(0)
            end if
        end function activation_derivderiv

        real(8) function logistic(x)
            real(8),intent(in) :: x
            logistic = 1.0d0/(1.0d0 + exp(-x))
        end function logistic

        real(8) function logistic_deriv(x)
            !* compute using logarithm to avoid 1/0 for large x
            
            real(8),intent(in) :: x

            !* scratch
            real(8) :: tmp
            tmp = exp(-x)
            logistic_deriv = exp(-(x+2.0d0*log(tmp+1.0d0)) )
        end function logistic_deriv

        real(8) function logistic_derivderiv(x)
            implicit none

            real(8),intent(in) :: x

            real(8) :: tmp1,tmp2,sgn

            tmp1 = exp(x)

            sgn = sign(1.0d0,tmp1-1.0d0)
            
            !* y''(x) = -sgn(e^x - 1) * e^x abs(e^x -1 )/(e^x+1)^3
            tmp2 = x + log(abs(tmp1-1.0d0)) - 3.0d0*log(tmp1+1.0d0)

            tmp2 = -sgn*exp(tmp2)
            
            if (isnan(tmp2)) then
                !* exp(x) not representable in double precision
                tmp2 = 0.0d0
            end if    
            logistic_derivderiv = tmp2
        end function logistic_derivderiv

        real(8) function tanh_deriv(x)
            real(8),intent(in) :: x
            !tanh_deriv = 1.0d0 - (tanh(x)**2)
            tanh_deriv = (1.0d0/cosh(x))**2
        end function tanh_deriv

        real(8) function tanh_derivderiv(x)
            implicit none

            real(8),intent(in) :: x

            real(8) :: tmp

            tmp = tanh(x)

            if (scalar_equal(tmp,0.0d0,dble(1e-10),dble(1e-16)**2,.false.)) then
                tanh_derivderiv = 0.0d0
                return
            end if

            tanh_derivderiv = -2.0d0*tmp/(cosh(x)**2)
        end function tanh_derivderiv
end module propagate
