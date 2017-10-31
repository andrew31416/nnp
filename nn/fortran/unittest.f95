program unittest

    use config
    use init
    use propagate
    use util
    use io
    use measures
    use features

    implicit none

    call main()

    contains
        subroutine main()
            implicit none

            logical :: tests(1:5)
            
            !* net params
            integer :: num_nodes(1:2),nlf_type,fD

            !* features
            integer :: natm,nconf

            !* scratch
            integer :: ii

            integer :: num_tests

            num_tests = size(tests)

            !* number of nodes
            num_nodes(1) = 3
            num_nodes(2) = 2

            !* nonlinear function
            nlf_type = 1
            
            !* features
            fD = 2
            natm = 2
            nconf = 2
            
            call unittest_header()
            
            !* generate atomic info
            call generate_testtrain_set(natm,nconf)

            !* initialise feature memory
            call generate_features(fD)

            !* calculate features and analytical derivatives
            call calculate_features()

            call initialise_net(num_nodes,nlf_type,fD)

            !----------------------!
            !* perform unit tests *!
            !----------------------!
            
            tests(1) = test_dydw()              ! dydw
            call test_loss_jac(tests(2:4))      ! d loss / dw
            tests(5) = test_dydx()               ! dydx


            do ii=1,num_tests
                call unittest_test(ii,tests(ii))    
            end do

            call unittest_summary(tests)

            if (all(tests)) then
                !* success
                call exit(1)
            else
                !* failure
                call exit(0)
            end if
        end subroutine main

        subroutine generate_testtrain_set(natm,nconf)
            implicit none

            integer,intent(in) :: natm,nconf

            !* scratch 
            integer :: set_type,conf,ii
            real(8) :: tmpz

            do set_type=1,2
                data_sets(set_type)%nconf = nconf
                allocate(data_sets(set_type)%configs(nconf))

                do conf=1,nconf
                    allocate(data_sets(set_type)%configs(conf)%r(3,natm))
                    allocate(data_sets(set_type)%configs(conf)%z(natm))
                    allocate(data_sets(set_type)%configs(conf)%current_ei(natm))
                    allocate(data_sets(set_type)%configs(conf)%current_fi(3,natm))
                    allocate(data_sets(set_type)%configs(conf)%forces(3,natm))
                    
                    data_sets(set_type)%configs(conf)%cell = 0.0d0
                    data_sets(set_type)%configs(conf)%cell(1,1) = 4.0d0
                    data_sets(set_type)%configs(conf)%cell(2,2) = 5.0d0
                    data_sets(set_type)%configs(conf)%cell(3,3) = 10.0d0

                    data_sets(set_type)%configs(conf)%n = natm
                    call random_number(data_sets(set_type)%configs(conf)%energy)
                    data_sets(set_type)%configs(conf)%forces = 0.0d0

                    do ii=1,3
                        call random_number(data_sets(set_type)%configs(conf)%r(ii,:))
                        data_sets(set_type)%configs(conf)%r(ii,:) = &
                                &data_sets(set_type)%configs(conf)%r(ii,:)*&
                                &data_sets(set_type)%configs(conf)%cell(ii,ii)
                    end do
                    
                    do ii=1,natm
                        if (ii.lt.natm/2) then
                            tmpz = 1.0d0
                        else
                            tmpz = 6.0d0
                        end if
                        data_sets(set_type)%configs(conf)%z(ii) = tmpz
                    end do
                end do
            end do
        end subroutine generate_testtrain_set

        subroutine generate_features(fD)
            implicit none

            integer,intent(in) :: fD

            integer :: ii,set_type,conf
            real(8) :: rcut

            allocate(feature_params%info(fD))
            feature_params%pca = .false.
            feature_params%pca_threshold = 0.0d0
            feature_params%num_features = fD


            rcut = 7.0d0
           
            feature_params%info(1)%ftype=0              !* atomic number
            do ii=2,feature_params%num_features
                feature_params%info(ii)%ftype = 1       !* behler-iso
                feature_params%info(ii)%rcut = rcut
                feature_params%info(ii)%fs = 0.2d0
                call random_number(feature_params%info(ii)%eta)
                call random_number(feature_params%info(ii)%za)
                call random_number(feature_params%info(ii)%zb)
            end do

            do set_type=1,2
                do conf=1,data_sets(set_type)%nconf
                    allocate(data_sets(set_type)%configs(conf)%x(fD+1,data_sets(set_type)%configs(conf)%n))
                    !allocate(data_sets(set_type)%configs(conf)%x_deriv(D+1,data_sets(set_type)%configs(conf)%n))
                end do
            end do
        end subroutine generate_features

        logical function test_dydw()
            implicit none

            logical :: test_result
    
            !* scratch
            integer :: ww,ii,jj,atm,conf,set_type
            real(8),allocatable :: original_weights(:)
            real(8),allocatable :: dydw_flat(:)
            real(8),allocatable :: derivs(:)
            real(8) :: dw,w0,dy,num_val

            allocate(original_weights(nwght))
            allocate(dydw_flat(nwght))
            allocate(derivs(nwght))

            !* initial weights
            call parse_structure_to_array(net_weights,original_weights)
            
            conf = 1
            set_type = 1
            atm = 1

            dy = 0.0d0

            do ww=4,4,1
                dw = 1.0d0/(10.0d0**ww)

                do ii=1,nwght,1
                    w0 = original_weights(ii)
                    do jj=1,2,1
                        if (jj.eq.1) then
                            original_weights(ii) = w0 + dw
                        else 
                            original_weights(ii) = w0 - dw
                        end if
                        
                        !* read in weights
                        call parse_array_to_structure(original_weights,net_weights)
                        
                        !* forward propagate
                        call forward_propagate(conf,atm,set_type)

                        if (jj.eq.1) then
                            dy = data_sets(set_type)%configs(conf)%current_ei(atm)
                        else 
                            dy = dy - data_sets(set_type)%configs(conf)%current_ei(atm)
                      end if
                    end do

                    num_val = dy/(2.0d0*dw)
 
                    derivs(ii) = num_val
               
                    !* return to initial value
                    original_weights(ii) = w0
                end do
               
                !--------------------------!
                !* analytical derivatives *!
                !--------------------------!

                call parse_array_to_structure(original_weights,net_weights)

                !* forward prop
                call forward_propagate(conf,atm,set_type)
                
                !* back prop
                call backward_propagate(conf,atm,set_type)
              
                !* parse dy/dw into 1d array
                call parse_structure_to_array(dydw,dydw_flat)
                
                test_result = array_equal(derivs,dydw_flat,dble(1e-15),dble(1e-10))
            end do
            test_dydw = test_result
        end function test_dydw

        subroutine test_loss_jac(test_result)
            !* test the jacobian of energy, force, reg. loss functions *!

            implicit none

            logical,intent(out) :: test_result(1:3)

            !* scratch
            integer :: ii,jj,kk,ww,conf,atm,set_type
            real(8) :: dw,w0,dloss,tmp
            real(8),dimension(:),allocatable :: num_jac,anl_jac,original_weights

            allocate(num_jac(nwght))
            allocate(anl_jac(nwght))
            allocate(original_weights(nwght))
            
            !* initial weights
            call parse_structure_to_array(net_weights,original_weights)

            conf= 1
            atm = 1
            set_type = 1

            dloss = 0.0d0

            do ii=1,3,1
                if (ii.eq.1) then
                    loss_const_energy = 1.0d0
                    loss_const_forces = 0.0d0
                    loss_const_reglrn = 0.0d0
                else if (ii.eq.2) then
                    loss_const_energy = 0.0d0
                    loss_const_forces = 1.0d0
                    loss_const_reglrn = 0.0d0
                else
                    loss_const_energy = 0.0d0
                    loss_const_forces = 0.0d0
                    loss_const_reglrn = 1.0d0
                end if
                
                do ww=4,4,1
                    !* finite difference
                    dw = 1.0d0/(10.0d0**(ww))

                    do jj=1,nwght,1
                        w0 = original_weights(jj)

                        do kk=1,2,1
                            if (kk.eq.1) then
                                original_weights(jj) = w0 + dw
                            else
                                original_weights(jj) = w0 - dw
                            end if
                            
                            tmp = loss(original_weights,nwght,set_type)

                            if (kk.eq.1) then
                                dloss = tmp
                            else
                                dloss = dloss - tmp
                            end if

                            original_weights(jj) = w0
                        end do !* end loop +/- dw

                        num_jac(jj) = dloss / (2.0d0 * dw)
                    end do !* end loop over weights

                end do !* end loop over dw

                !-------------------------------!
                !* analytical jacobian of loss *!
                !-------------------------------!

                call loss_jacobian(original_weights,nwght,set_type,anl_jac)
                
                test_result(ii) = array_equal(num_jac,anl_jac,dble(1e-15),dble(1e-10))
            end do !* end loop over loss terms
        end subroutine test_loss_jac

        logical function test_dydx()
            implicit none

            !* scratch 
            integer :: set_type,conf,atm,xx,ii,ww
            real(8) :: dw,x0
            real(8),allocatable :: num_dydx(:)
            logical,allocatable :: log_atms(:),cnf_atms(:)
            logical :: set_atms(1:2)

            allocate(num_dydx(D))

            do set_type=1,2
                allocate(cnf_atms(data_sets(set_type)%nconf))

                do conf=1,data_sets(set_type)%nconf
                    allocate(log_atms(data_sets(set_type)%configs(conf)%n))
                    
                    do atm=1,data_sets(set_type)%configs(conf)%n

                        !--------------------------!
                        !* numerical differential *!
                        !--------------------------!
                        
                        do xx=1,D
                           x0 = data_sets(set_type)%configs(conf)%x(xx+1,atm)

                            do ww=5,5 
                                !* finite difference for feature
                                dw = 1.0d0/(10**ww)

                                do ii=1,2
                                    if (ii.eq.1) then
                                        data_sets(set_type)%configs(conf)%x(xx+1,atm) = x0 + dw
                                    else
                                        data_sets(set_type)%configs(conf)%x(xx+1,atm) = x0 - dw
                                    end if
                                    
                                    call forward_propagate(conf,atm,set_type)
   
                                    if (ii.eq.1) then
                                        num_dydx(xx) = data_sets(set_type)%configs(conf)%current_ei(atm)
                                    else
                                        num_dydx(xx) = num_dydx(xx) - &
                                                &data_sets(set_type)%configs(conf)%current_ei(atm)
                                    end if
                                    
                                    data_sets(set_type)%configs(conf)%x(xx+1,atm) = x0 
                                end do !* end loop +/- dw

                                num_dydx(xx) = num_dydx(xx) / (2.0d0*dw)
                            end do !* end loop finite differences
                        end do !* end loop features
                        
                        !---------------------------!
                        !* analytical differential *!
                        !---------------------------!
                        
                        call forward_propagate(conf,atm,set_type)
                        call backward_propagate(conf,atm,set_type)
                    
                        log_atms(atm) = array_equal(num_dydx,dydx,dble(1e-15),dble(1e-10))

                    end do !* end loop atoms
                
                    cnf_atms(conf) = all(log_atms)

                    deallocate(log_atms)
                end do !* end loop configurations

                set_atms(set_type) = all(cnf_atms)

                deallocate(cnf_atms)
            end do !* end loop data sets

            test_dydx = all(set_atms)
        end function test_dydx

end program unittest