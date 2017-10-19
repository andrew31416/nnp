module measures
    use propagate
    use util

    implicit none

    real(8),external :: dnrm2

    real(8) :: tot_energy_loss
    real(8) :: tot_forces_loss
    real(8) :: tot_reg_loss

    type(weights),private :: loss_deriv 

    contains

        real(8) function loss(flat_weights,nw,set_type)
            
            implicit none

            integer,intent(in) :: nw,set_type
            real(8),intent(in) :: flat_weights(1:nw)

            !* scratch
            integer :: conf,atm,loss_type
            real(8) :: const_energy,const_forces,const_reg

            !* read in NN weights
            call parse_weights_expand(flat_weights,nw)

            !* energy loss constant
            const_energy = 1.0d0
            
            !* forces loss constant
            const_forces = 1.0d0

            !* regularization constant
            const_reg = 1.0d0

            !* type of norm for loss
            loss_type = 2

            do conf=1,data_sets(set_type)%nconf,1
                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    !* forward prop on training data
                    call forward_propagate(conf,atm,1)
                end do
            end do

            call loss_energy(set_type,loss_type)
            call loss_forces(set_type,loss_type)
            call loss_regularization(flat_weights)
            
            loss = tot_reg_loss*const_reg +&
                    &tot_forces_loss*const_forces +&
                    &tot_energy_loss*const_energy
        end function loss

        subroutine loss_derivative(flat_weights,nw,set_type)

            implicit none
            
            integer,intent(in) :: nw,set_type
            real(8),intent(in) :: flat_weights(1:nw)
            
            !* scratch
            integer :: conf,atm

            do conf=1,data_sets(set_type)%nconf,1
                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    !* forward prop on training data
                    call forward_propagate(conf,atm,1)
                    
                    call backward_propagate(conf,atm,set_type)
                end do
            end do

        end subroutine loss_derivative

        subroutine loss_energy(set_type,loss_type)
            implicit none
            
            integer,intent(in) :: set_type,loss_type

            !* scratch
            real(8) :: tmp1,tmp2
            integer :: conf

            tot_energy_loss = 0.0d0

            do conf=1,data_sets(set_type)%nconf,1
    
                !* 1/Natm
                tmp2 = 1.0d0/dble(data_sets(set_type)%configs(conf)%n)

                !* consider energy per atom
                tmp1 = abs(sum(data_sets(set_type)%configs(conf)%current_ei)&
                        & - data_sets(set_type)%configs(conf)%energy) * tmp2
                
                if (loss_type.eq.1) then
                    !* l1 norm
                    tot_energy_loss = tot_energy_loss + tmp1
                else if (loss_type.eq.2) then
                    !* l2 norm
                    tot_energy_loss = tot_energy_loss + tmp1**2
                end if

            end do
            
            tot_energy_loss = tot_energy_loss * 0.5d0 
        end subroutine loss_energy

        subroutine loss_forces(set_type,loss_type)
            implicit none

            integer,intent(in) :: set_type,loss_type

            !* scratch
            integer :: conf,atm,ii
            real(8) :: tmp

            tot_forces_loss = 0.0d0

            !* could fill 1-d arrays and use lapack here

            do conf=1,data_sets(set_type)%nconf,1
                do atm=1,data_sets(set_type)%configs(conf)%n,1
                    do ii=1,3,1
                        tmp = abs(data_sets(set_type)%configs(conf)%current_fi(ii,atm) - &
                        & data_sets(set_type)%configs(conf)%forces(ii,atm))

                        if (loss_type.eq.1) then
                            tot_forces_loss = tot_forces_loss + tmp
                        else
                            tot_forces_loss = tot_forces_loss + tmp**2
                        end if
                    end do
                end do
            end do

            tot_forces_loss = tot_forces_loss * 0.5d0
        end subroutine loss_forces

        subroutine loss_regularization(flat_weights)
            implicit none

            real(8),intent(in) :: flat_weights(:)

            !* l2 norm
            tot_reg_loss = dnrm2(size(flat_weights),flat_weights,1) * 0.5d0
        end subroutine loss_regularization


        subroutine loss_energy_deriv()
            implicit none

            !* calculate dy/dw
        end subroutine
end module measures
